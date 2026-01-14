
import unittest
from unittest.mock import MagicMock, patch, AsyncMock, ANY
import sys
import os
import types

# --- 1. Aggressive Mocking of Missing Modules ---
# We must mock dependencies BEFORE importing pages.e2e or AI_pipeline_general

def mock_module(module_name):
    parts = module_name.split('.')
    for i in range(1, len(parts) + 1):
        name = '.'.join(parts[:i])
        if name not in sys.modules:
            m = MagicMock()
            m.__path__ = [] 
            m.__spec__ = MagicMock()
            m.__file__ = f"mock://{name}"
            sys.modules[name] = m
        
        if i > 1:
            parent_name = '.'.join(parts[:i-1])
            child_name = parts[i-1]
            parent = sys.modules[parent_name]
            setattr(parent, child_name, sys.modules[name])

modules_to_mock = [
    "streamlit",
    "rich",
    "rich.console",
    "yaml",
    "neo4j",
    "langchain_ollama",
    "langchain_neo4j",
    "google",
    "google.genai",
    "mcp",
    "langgraph",
    "langgraph.graph",
    "langgraph.graph.message",
    "langgraph.checkpoint.memory",
    "AI_pipeline_general" # We mock this one too to control the Agent
]

for mod in modules_to_mock:
    mock_module(mod)

# Mock SessionState to support both attribute and item access
class MockSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

sys.modules["streamlit"].session_state = MockSessionState()
sys.modules["streamlit"].columns.return_value = (MagicMock(), MagicMock())
# Make st.empty().status() a context manager
mock_status = MagicMock()
sys.modules["streamlit"].empty.return_value.status.return_value.__enter__.return_value = mock_status

# Setup AI_pipeline_general mock
mock_pipeline = sys.modules["AI_pipeline_general"]
mock_pipeline.Langgraph = MagicMock()

# Ensure we can import pages.e2e eventually
# pages/e2e.py expects project_root to be set up.
# We will execute the file content directly, so we don't need 'pages' package in sys.modules necessarily,
# but the script does imports.

class TestE2EHumanLoop(unittest.TestCase):

    def setUp(self):
        # Reset streamlit session state before each test
        sys.modules["streamlit"].session_state = MockSessionState()
        # Reset mocks
        sys.modules["streamlit"].reset_mock()
        sys.modules["AI_pipeline_general"].reset_mock()
        
        # Path to tests_new.py (formerly e2e.py)
        self.script_path = os.path.join(os.getcwd(), "pages", "tests_new.py")
        with open(self.script_path, "r") as f:
            self.script_content = f.read()

    def test_interruption_tester_node(self):
        print("\n--- Testing Human-in-the-Loop: Tester Node Interruption ---")
        
        # 1. Setup Streamlit Mocks for this scenario
        st = sys.modules["streamlit"]
        
        # Mock Session State Inputs
        st.session_state["messages"] = []
        
        # Mock Agent
        mock_agent = MagicMock()
        st.session_state["agent"] = mock_agent
        st.session_state["thread_id"] = "test_thread"
        
        # Mock Graph Snapshot (Paused at Tester)
        mock_snapshot = MagicMock()
        mock_snapshot.next = ("Tester",)
        mock_snapshot.values = {"instructions": "Original Instructions"}
        mock_agent.graph.get_state.return_value = mock_snapshot
        
        # Mock UI Inputs
        # user_input = None (don't trigger first if block)
        st.chat_input.return_value = None
        
        # auto_mode = False
        # The script calls: auto_mode = st.checkbox(...)
        # We need to control what checkbox returns.
        # Since text inputs and checkboxes are called in order, we can use side_effect or try to key off label.
        # But for simpler mocking, let's just make all checkboxes False (default)
        st.checkbox.return_value = False
        
        # Mock "Run test" button to be True (User clicked it)
        # col1.button("Run test") -> True
        # st.columns returns (col1, col2)
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        st.columns.return_value = (mock_col1, mock_col2)
        mock_col1.button.side_effect = lambda label: True if label == "Run test" else False
        
        # Mock "Write here" text input for new instructions
        st.text_input.side_effect = lambda label, value=None: "New Instructions" if "Write here" in label else "default"
        
        # Mock Agent Graph updates
        # The script calls asyncio.run(run_interaction(resume_data={"new_instructions":...}))
        # which calls agent.graph.update_state and agent.graph.astream
        
        # We need to handle asyncio.run. 
        # Since we are mocking it, we can just let it call our sync mock or just verify it was called.
        # Use patch for asyncio only for this test
        with patch("asyncio.run", MagicMock()) as mock_async_run:
             # Run the script
            exec(self.script_content, globals())
            
            # Assertions
            print("Verifying 'Run test' logic...")
            
            # 1. Verify we retrieved the state
            mock_agent.graph.get_state.assert_called()
            
            # 2. Verify we checked for 'Tester' step
            # Implied by falling into the if block.
            
            # 3. Verify asyncio.run was called to trigger run_interaction
            # The script defines run_interaction inside. 
            # We can't easily mock the internal function 'run_interaction' before it's defined.
            # But we can verify asyncio.run was called.
            self.assertTrue(mock_async_run.called, "asyncio.run should be called")
            
            # To be more robust, we want to know IF update_state was called.
            # But update_state is called INSIDE run_interaction which is an async func called by asyncio.run.
            # If we mocked asyncio.run, the coroutine passed to it was NOT executed!
            # So update_state won't technically be called unless we manually await the coroutine passed to mocked asyncio.run.
            
            # Let's see what was passed to asyncio.run
            # args[0] is the coroutine object 'run_interaction(...)'
            coro = mock_async_run.call_args[0][0]
            
            # We can inspect the coroutine's locals? No.
            # We should probably run the coroutine if possible, or Mock asyncio.run to accept a coro and run it?
            pass

    def test_run_interaction_execution(self):
        # This test focuses on refining the 'asyncio.run' mock to actually execute the logic
        # so we can verify agent calls.
        print("\n--- Testing run_interaction Logic ---")
        
        st = sys.modules["streamlit"]
        st.session_state["messages"] = []
        mock_agent = MagicMock()
        st.session_state["agent"] = mock_agent
        st.session_state["thread_id"] = "test_thread"
        mock_snapshot = MagicMock()
        mock_snapshot.next = ("Tester",)
        mock_snapshot.values = {"instructions": "Original"}
        mock_agent.graph.get_state.return_value = mock_snapshot
        
        st.chat_input.return_value = None
        st.checkbox.return_value = False # auto_mode off
        
        mock_col1 = MagicMock()
        st.columns.return_value = (mock_col1, MagicMock())
        mock_col1.button.return_value = True # Click Run test
        
        # We need a proper async runner for the mock
        # We need to make astream an async iterator
        async def async_gen():
            yield {"Tester": {"messages": [("assistant", "Output")]}}
        
        mock_agent.graph.astream.side_effect = lambda *args, **kwargs: async_gen()
        
        # We DO NOT patch asyncio.run here. We let the real asyncio run the coroutine.
        # Since run_interaction calls agent.graph.astream, which we mocked as async gen, it should work.
        
        exec(self.script_content, globals())
             
        # Now agent.graph.update_state SHOULD have been called
        mock_agent.graph.update_state.assert_called()
        call_args = mock_agent.graph.update_state.call_args
        # args: (config, resume_data)
        resume_data = call_args[0][1] # second positional arg
         
        print(f"Update State called with: {resume_data}")
        self.assertIn("new_instructions", resume_data)
         
        # Verify graph.astream was called
        mock_agent.graph.astream.assert_called()
        print("Verified agent execution triggered.")

if __name__ == '__main__':
    unittest.main()
