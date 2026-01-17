from typing import Annotated,Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os,sys,traceback
from dotenv import load_dotenv
# Explicitly load .env from the script's directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)
import json
import asyncio
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
import re
from rich.console import Console
from rich.panel import Panel
import datetime
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp_server import MCPPlaywright,MCPNeo4J,generator

def clean_code_block(text: str) -> str: #gets only the acc code inside the block
    pattern = r"```(?:python)?\n(.*?)```"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return text.strip()

def location(results:bool,test_type:str)->str:
    if results: #if its a test file
        path="./results"
    else: #if its a code block
        if test_type == "Parameter":
            path = "./tests/codeblock/param"
        else:
            path = f"./tests/{test_type}"
    
    if results:
        new_path = os.path.join(path, test_type)
    else:
        new_path = path
    
    return new_path

class State(TypedDict): #create message history
    messages: Annotated[list,add_messages]
    vector_store: Neo4jVector
    tool_history: list[dict]
    filename:str
    instructions:str|None
    new_instructions:str|None

class Langgraph:
    def __init__(self,test_type:str,neo4j_url:str,neo4j_database:str,neo4j_pwd:str,neo4j_ai_model:str,app_address:str,max_AI_steps:int,headless:bool,tester_ai:str,code_generator_ai:str,similarity_k:int=20,columns:list[str]=[],csv_path:str=None):
        """
        Docstring for __init__
        :param test_type: Type of test you are running (e.g. E2E etc, will change the retrieval query)
        :type test_type: str

        :param neo4j_url: URL for your neo4j database
        :type neo4j_url: str

        :param neo4j_pwd: Password for your neo4j database
        :type neo4j_pwd: str

        :param neo4j_database: database name for your neo4j database
        :type neo4j_database: str

        :param neo4j_ai_model: AI model if you wanna set for your neo4j AI model
        :type neo4j_ai_model: str

        :param app_address: The address or link for the agent to access your website
        :type app_address: str
        
        :param max_AI_steps: Max number of steps that your tester agent can do before it just gives up
        :type max_AI_steps: int

        :param headless: Run the code in headless mode or not
        :type headless: bool
        
        :param tester_ai: The AI model for your tester agent
        :type tester_ai: str
        
        :param code_generator_ai: The AI model for your code generator agent
        :type code_generator_ai: str
        
        :param similarity_k: The number of nodes that you wanna retrieve during graphRAG. MIGHT NEED TO CHANGE TO DEPEND ON test_type
        :type similarity_k: int

        :param columns: The columns of the dataframe being passed in
        :type columns: list[str]
        
        :param csv_path: Path to the CSV file for parameter testing
        :type csv_path: str
        """


        self.messages=[]
        self.neo4j_url= neo4j_url
        self.test_type=test_type
        self.neo4j_pwd= neo4j_pwd
        self.neo4j_database = neo4j_database
        self.app_address=app_address
        self.max_AI_steps = max_AI_steps
        self.headless = headless
        self.neo4j_ai=neo4j_ai_model
        self.tester_ai=tester_ai
        self.code_generator_ai = code_generator_ai
        self.csv_path = csv_path
        self.column_names = columns
        self.expected_results = [f"expected_response_{col}" for col in columns]
        if csv_path:
            df= pd.read_csv(csv_path)
            try:
                self.first_col=df.iloc[0]
            except:
                self.first_col="User didnt provide data"

        # Performance optimization: cache connections and checks
        self.vector_store_cache = None
        self.neo4j_driver = None
        self._embeddings_checked = False

        #this is general in case we have others
        self.retrieval_query = """
                RETURN 
                    coalesce(node.CODE, '') AS text,
                    score,
                    {
                        id: elementId(node),
                        name:coalesce(node.NAME, node.FULL_NAME, 'Unnamed'),
                        labels: labels(node),
                        code: node.CODE
                    } AS metadata
                """
        self.tester_sys_prompt=f"""
                You are a useful agent who can use the browser using your tools in order to carry out instructions.
                The data you will be testing are {[col for col in columns] if columns else "None"}
                You must input each of those data into their respective fields
                RULES:
                1. Read the instructions provided in the input.
                2. Execute the steps sequentially using the browser tools.
                3. Do NOT answer from memory; use the tools.
                4. If the instruction is to click, use `browser_click`.
                5. CRITICAL: Once you have performed the requested actions and verified the result visually in the DOM, call `browser_close`.
                6. AFTER calling `browser_close`, DO NOT ATTEMPT TO RE-OPEN THE BROWSER.
                7. DO NOT try to verify the test again using `browser_run_code` or JavaScript injection. One pass is enough.
                8. Once the browser is closed, simply output the final report.

                Give a response in the following format:
                Test title:
                Test success: True/False
                Test explanation:
                Steps taken and following results:
                """
        self.graph_sys_prompt=f"""
            You are a graph-based testing expert.

            IMPORTANT RULES:
            - You MUST use tools to inspect the graph before answering.
            - Do NOT answer from memory.
            - If information is missing, explore the graph using tools.
            - Only produce a final answer AFTER tool usage.

            MENTION THE APP WILL BE OPENED ON {self.app_address}

            Output format:
            Path_exists: True/False
            test_steps:
            - step: 1
            action: navigate
            instruction: ...
            target: ...
            expected: ...
            """
        self.generate_code_system_prompt="""
        You are a Senior QA Automation Engineer.
        Convert the following execution history into a **Pytest-Playwright** test file.
        
        RULES:
        1. **Structure**: Use the standard `def test_scenario(page: Page):` format.
        2. **Assertions**: A test is meaningless without checks. You MUST include `expect()` assertions.
        - If there is a clicked a button that increments a counter, assert the new text (e.g., `expect(button).to_contain_text(...)`).
        - If the user navigated, assert the URL or page title.
        - Whatever the user does, convert it into a test file
        3. **Cleanup**: Remove redundant steps (like repeated navigations).
        4. **Syntax**: 
        - `from playwright.sync_api import Page, expect`
        - Use `page.get_by_role` or `page.locator` with robust regex selectors.
        5. **Regex**: When using Regex selectors in Python, you MUST import re and use re.compile(r'pattern'). Do NOT pass raw regex strings.
        6. **Waiting**: When working with browser functions, make sure to wait for anything that causes loading, such as opening links or causing page switches


        Use AI
        Output ONLY the python code block.
        
        """

        if test_type == "E2E":
            self.retrieval_query = """
                    RETURN 
                        coalesce(node.CODE, '') AS text,
                        score,
                        {
                            id: elementId(node),
                            name:coalesce(node.NAME, node.FULL_NAME, 'Unnamed'),
                            labels: labels(node),
                            code: node.CODE
                        } AS metadata
                    """                
            self.tester_sys_prompt="""
                You are a useful agent who can use the browser using your tools in order to carry out instructions.
                
                RULES:
                1. Read the instructions provided in the input.
                2. Execute the steps sequentially using the browser tools.
                3. Do NOT answer from memory; use the tools.
                4. If the instruction is to click, use `browser_click`.
                5. CRITICAL: Once you have performed the requested actions and verified the result visually in the DOM, call `browser_close`.
                6. AFTER calling `browser_close`, DO NOT ATTEMPT TO RE-OPEN THE BROWSER.
                7. DO NOT try to verify the test again using `browser_run_code` or JavaScript injection. One pass is enough.
                8. Once the browser is closed, simply output the final report.

                Give a response in the following format:
                Test title:
                Test success: True/False
                Test explanation:
                Steps taken and following results:
                """
            self.graph_sys_prompt=f"""
            You are a graph-based testing expert.

            IMPORTANT RULES:
            - You MUST use tools to inspect the graph before answering.
            - Do NOT answer from memory.
            - If information is missing, explore the graph using tools.
            - Only produce a final answer AFTER tool usage.

            MENTION THE APP WILL BE OPENED ON {self.app_address}

            Output format:
            Path_exists: True/False
            test_steps:
            - step: 1
            action: navigate
            instruction: ...
            target: ...
            expected: ...
            """
            self.generate_code_system_prompt="""
                You are a Senior QA Automation Engineer.
                Convert the following execution history into a **Pytest-Playwright** test file.
                
                RULES:
                1. **Structure**: Use the standard `def test_scenario(page: Page):` format.
                2. **Assertions**: A test is meaningless without checks. You MUST include `expect()` assertions.
                - If the user clicked a button that increments a counter, assert the new text (e.g., `expect(button).to_contain_text(...)`).
                - If the user navigated, assert the URL or page title.
                3. **Cleanup**: Remove redundant steps (like repeated navigations).
                4. **Syntax**: 
                - `from playwright.sync_api import Page, expect`
                - Use `page.get_by_role` or `page.locator` with robust regex selectors.
                5. **Regex**: When using Regex selectors in Python, you MUST import re and use re.compile(r'pattern'). Do NOT pass raw regex strings.
                6. **Waiting**: When working with browser functions, make sure to wait for anything that causes loading, such as opening links or causing page switches

                Output ONLY the python code block.
                
                """


        if test_type == "Parameter": #find all the forms
            column_display = ", ".join(columns) if columns else "(no columns loaded yet)"
            expected_display = ", ".join([f"expected_response_{col}" for col in columns]) if columns else "(no columns loaded yet)"
            if hasattr(self,"first_col") and hasattr(self.first_col, "get"):
                test_data = " , ".join(f"{col}:{self.first_col.get(col,"N/A")}" for col in columns)
            else:
                test_data = "No test data available (CSV load failed or empty)"
            self.retrieval_query = """
            // 1. ZOOM OUT to Component Root
        OPTIONAL MATCH (node)<-[:AST|CONTAINS*0..20]-(m:METHOD)
        WITH node, score, collect(DISTINCT m) AS methods
        WITH coalesce(head(methods), node) AS root, score

        // 2. GATHER ALL UNIQUE CHILDREN FIRST (Including Routes)
        MATCH (root)-[:CONTAINS|AST*]->(child)
        WHERE 
        // HTML Elements
            (
                child.NAME IN ['a', 'Link', 'img', 'Image', 'input', 'button'] 
                OR child.CODE STARTS WITH '<a' 
                OR child.CODE STARTS WITH '<img' 
                OR child.CODE STARTS WITH '<button' 
                OR child.CODE STARTS WITH '<input'
                OR child.NAME IN ['push', 'navigate', 'redirect', 'go', 'back']
            )
            AND NOT child.NAME IN ['JSXOpeningElement', 'JSXClosingElement']

        WITH root, score, child.CODE as code, head(collect(child)) as unique_node
        
        // 4. COLLECT THE UNIQUE NODES INTO A LIST
        WITH root, score, collect(unique_node) as unique_children

        // 4. CATEGORIZE
        RETURN
            root.CODE as text,
            score,
            {
                id: elementId(root),
                name: root.NAME,
                labels: labels(root),
                
                links: [c IN unique_children 
                        WHERE c.NAME IN ['a', 'Link'] OR c.CODE STARTS WITH '<a' 
                        | {id: elementId(c), code: c.CODE}],

                images: [c IN unique_children 
                        WHERE c.NAME IN ['img', 'Image'] OR c.CODE STARTS WITH '<img' 
                        | {id: elementId(c), code: c.CODE}],
                
                inputs: [c IN unique_children 
                        WHERE c.NAME IN ['input'] OR c.CODE STARTS WITH '<input' 
                        | {id: elementId(c), code: c.CODE}],
                
                buttons: [c IN unique_children 
                        WHERE c.NAME IN ['button'] OR c.CODE STARTS WITH '<button' 
                        | {id: elementId(c), code: c.CODE}],
                
                routes: [c IN unique_children 
                        WHERE c.NAME IN ['push', 'navigate', 'redirect', 'go', 'back'] 
                        | {id: elementId(c), name: c.NAME, code: c.CODE}]
            } as metadata"""       
            self.tester_sys_prompt=f"""
            You are a useful parameter tester. You will be tasked with finding WHERE the location is for testing the functionality of certain components in the website
            DATA TO TEST: {", ".join(columns) if columns else "None"}

                RULES:
                1. Read the instructions provided in the input.
                2. Execute the steps sequentially using the browser tools.
                3. Do NOT answer from memory; use the tools.
                4. When using tools
                - Locate the corresponding input field on the page
                - Note its ID, name, and unique selector
                - Fill it with the test value from that column: {test_data}
                - Once filled any data, if it is part of a form, submit the form if requested by user
                5. If the instruction is to click, use `browser_click`.
                6. Once all instructions are done, check for a response if response is expected, and note down the ID of what displays a response from the website
                7. CRITICAL: Once you have performed the requested actions, call `browser_close`.
                8. AFTER calling `browser_close`, DO NOT ATTEMPT TO RE-OPEN THE BROWSER.
                9. DO NOT try one pass is enough.
                10. Once the browser is closed, simply output the final report.

                Give a response in the following format: (STRICT)
                CRITICAL SELECTOR RULES:
                - When multiple elements match a label (like "Email Address"), you MUST use the most specific selector
                - Prefer ID selectors: Use page.locator("#specific-id") instead of get_by_label() when ambiguous
                - For forms, identify what form and use the correct ID
                - Example: If there are #login-email and #reg-email, determine which form you're testing and use that specific ID
                - ALWAYS record the EXACT selector (with ID) you used in your output, not just the label
    
                OUTPUT FORMAT (STRICT):
                Field Mappings (use EXACT selectors like page.locator("#reg-email") or page.locator("#field-id")):
                {chr(10).join(f"- {col}: <exact_selector_with_id>" for col in columns) if columns else "- field1: <exact_selector_with_id>"}
            
            """ 
            self.graph_sys_prompt=f"""
                You are a graph-based form testing expert specializing in parameter validation.

                TASK: Find the form/component that accepts these inputs: {", ".join(columns) if columns else "None"}

                RULES:
                - Use tools to inspect the Neo4j graph
                - Find form components, input fields, and submit buttons
                - Match field names/IDs to the input columns: {", ".join(columns) if columns else "None"}
                - Identify where results/responses appear after submission
                - Be as specific as possible and give the exact IDs or unique selectors for each field
                - For the test instructions, use this dummy data: {test_data}
                APP URL: {self.app_address}

                SELECTOR DISAMBIGUATION:
                - Use ID for almost everything, or anything that can help a playwright locator. Try not to use just the text thats seen but the html ids etc. 
                - The more specific to that specific field the better
                - Prioritize form-specific context (e.g., registration form vs login form)
                - Include the full unique selector (ID, data-testid, or unique ancestor path) in your output

                OUTPUT FORMAT:
                Path_exists: True/False

                Form_location: <component_name or route>

                Field_mappings:
                - {columns[0] if columns else 'field1'}: <field_selector>
                - {columns[1] if columns and len(columns) > 1 else 'field2'}: <field_selector>

                Result_location: <where_response_appears>

                INSTRUCTIONS: (example)
                Test_steps:
                - step: 1
                action: navigate
                instructions: ...
                target: {self.app_address}
                expect: .....
                
                - step: 2
                action: fill_form
                instructions: ...
                target: .... MAKE SURE TO USE THE IDs IF POSSIBLE OR MOST UNIQUE DATA
                expect:....

                - step: 3
                action: submit
                instructions:...
                target: ...
                expect:....

                """
            self.generate_code_system_prompt=f"""
            You are a Senior QA Automation Engineer generating PARAMETERIZED test code.

            CONTEXT:
            - CSV Path: {self.csv_path}
            - Input Columns: {column_display}
            - Expected Columns: {expected_display}

            SELECTOR BEST PRACTICES:
            - NEVER use ambiguous selectors like get_by_label() if multiple elements match
            - ALWAYS use specific IDs: page.locator("#reg-email") not page.get_by_label("Email Address")
            - From the execution history, extract the EXACT selectors that worked during testing
            - If a field has an ID attribute, try locating through ID
            - Chain locators when needed: page.locator("#registration-form").get_by_label("Email")
            - Look at the tool_history for browser_click, browser_type, and browser_fill_form calls - these contain the actual selectors used
            - Extract selectors from successful interactions in the execution history

            CRITICAL REQUIREMENTS:
            1. Access CSV data using: row["column_name"]
            2. Access expected results using: row["expected_response_column_name"]
            3. For EACH input column, you must:
            - Get the value: value = row["column_name"]
            - Fill the corresponding field using the selector from execution history
            - Example: await page.fill("#email-input", row["email"])

            4. After filling all fields:
            - Click the submit button
            - Wait for response/navigation: await page.wait_for_load_state("networkidle")
            - Extract the actual result from the page

            5. Compare actual vs expected for EACH column:
            - expected = row["expected_response_column_name"]
            - assert actual == expected, f"Expected {{expected}}, got {{actual}}"

            6. DO NOT include: imports, CSV reading loop, or main function
            7. Output ONLY the indented test logic (inside the try block)
            8. Use proper async/await syntax
            9. Add meaningful wait statements after actions that trigger loading

            GENERATE EXACTLY THIS CODE, BUT THE ONLY CODE YOU WILL ADD IS BETWEEN THE TRY BLOCK
            ```python
            import pandas as pd
            import os
            import asyncio
            from playwright.async_api import async_playwright, Page, expect

            csv_path = r"{self.csv_path}"
            df = pd.read_csv(csv_path)

            print(f"Running {{len(df)}} test cases from CSV")

            async def main():
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch(headless={str(self.headless)})
                    for index, row in df.iterrows():
                        print(f"\\\\n=== Test Case {{index + 1}}/{{len(df)}} ===\")
                        print(f"Input values: {{dict(row)}}")
                        context = await browser.new_context()
                        page = await context.new_page()        
                        try:
                            # >>> YOUR CODE GOES HERE <
                        except Exception as e:
                            print(f"Test case {{index + 1}} FAILED: {{e}}")
                        else:
                            print(f"Test case {{index + 1}} PASSED")
                        finally:
                            await context.close()
                        except Exception as e:
                            print(f"Test case {{index + 1}} FAILED: {{e}}")
                        else:
                            print(f"Test case {{index + 1}} PASSED")
                        finally:
                            context.close()
                    await browser.close()
        if __name__ == "__main__":
            asyncio.run(main())
            print("\\\\nAll tests completed!")
            ```

            OUTPUT REQUIREMENTS:
            - Must be valid Python with proper indentation
            - Must use async/await for all Playwright calls
            - Must include assertions for every expected result column
            - Must handle waits properly
            """

        
        self.similarty_k = similarity_k
        self.embedding_uri="bolt://localhost:7687"
        self.embedding_auth=("neo4j", "password")
        self.memory=MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        def check_embeddings() -> bool:
            # Performance optimization: use cached result if already checked
            if self._embeddings_checked:
                print("DEBUG: Using cached embedding validation result.")
                return True
            
            """ Check if embeddings are there or not"""
            driver = GraphDatabase.driver(self.embedding_uri, auth=self.embedding_auth)
            
            try:
                with driver.session() as session:
                    index_exists_query = """
                    SHOW INDEXES YIELD name, type
                    WHERE name = 'general_components' AND type = 'VECTOR'
                    RETURN count(*) > 0 AS exists
                    """
                    index_result = session.run(index_exists_query).single()
                    
                    if not index_result or not index_result["exists"]:
                        print("DEBUG: Vector index 'general_components' missing.")
                        return False

                    data_exists_query = """
                    MATCH (n:GeneralComponent)
                    WHERE n.embedding_general IS NOT NULL 
                    RETURN count(n) > 0 AS has_data LIMIT 1
                    """
                    data_result = session.run(data_exists_query).single()
                    
                    if not data_result or not data_result["has_data"]:
                        print("DEBUG: Index exists, but no nodes have 'embedding_general' property.")
                        return False
                        
                    print("DEBUG: Embeddings verified (Index + Data found).")
                    
                    # Cache the successful result
                    self._embeddings_checked = True
                    return True

            except Exception as e:
                print(f"DEBUG: Error checking embeddings: {e}")
                return False
            finally:
                driver.close()

        def LabelSetupNode(state:State):
            driver = GraphDatabase.driver(self.embedding_uri, auth=self.embedding_auth)
            print("Creating GeneralComponent label to certain nodes..")
            query = """
            MATCH (n) 
            WHERE any(l IN labels(n) WHERE l IN [
                'TEMPLATE_DOM', 'METHOD', 'CALL', 'IDENTIFIER', 
                'LITERAL', 'TYPE_DECL', 'METHOD_PARAMETER_IN', 'METHOD_PARAMETER_OUT'
            ])
            SET n:GeneralComponent
            """
            # If need to be more general, just add the labels here 

            with driver.session() as session:
                session.run(query)
                print("Verified GeneralComponent labels")
            
            return state

        def EmbeddingNode(state:State):
                """
                Takes the entire graph and embed it
                """
                neo4j_url = "bolt://localhost:7687"
                neo4j_password = "password"

                embeddings = OllamaEmbeddings(
                model="nomic-embed-text:latest",
                base_url="http://localhost:11434"
                )

                # creating the embeddings
                vector_store = Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=neo4j_url,
                    password=neo4j_password,
                    index_name="general_components",
                    node_label="GeneralComponent",
                    text_node_properties=["NAME", "CODE"],
                    embedding_node_property="embedding_general",
                    retrieval_query=self.retrieval_query
                )
                
                # return {"vector_store": vector_store
                return state

        def VectorSearchNode(state:State):
            print("Doing vector search..")
            
            # Performance optimization: reuse cached vector store
            store = self.vector_store_cache
            
            if not store:
                print("    (Connecting to Neo4j index for first time...) ")
                neo4j_url = self.neo4j_url
                neo4j_password = self.neo4j_pwd
                embeddings = OllamaEmbeddings(
                    model="nomic-embed-text:latest", 
                    base_url="http://localhost:11434"
                )
                
                # connects to the DB.
                store = Neo4jVector.from_existing_index(
                    embedding=embeddings,
                    url=neo4j_url,
                    password=neo4j_password,
                    index_name="general_components",
                    retrieval_query=self.retrieval_query
                )
                
                # Cache for future queries - critical performance optimization!
                self.vector_store_cache = store
            else:
                print("    (Reusing cached vector store connection)")
            
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                # if an object
                user_query = last_message.content
            else:
                # a tuple ("user", "query")
                user_query = last_message[1]

            print(f"Querying for {user_query}..")
            k=self.similarty_k
            results = store.similarity_search_with_score(user_query, k=k) # change k depending on how many nodes you want to return

            if not results:
                text = "Component not found"
                print(text)
                return {"messages": [("assistant", text)]}

            components = []
            # Performance optimization: use higher threshold for parameter testing
            threshold = 0.80 if self.test_type == "Parameter" else 0.70
            
            for document, score in results:
                if score < threshold: # stricter filtering for parameter tests
                    continue
                meta = document.metadata
                item_str = f"Name: {meta.get('name', 'Unnamed')} | ID: {meta.get('id')} | Code: {meta.get('code')} | Score: {score:.4f}"
                components.append(item_str)

            final_response = "\n".join(components)
            if not components:
                final_response = "No components found with high enough confidence."
            return {"messages": [("assistant",final_response)]}

        async def MCPGraph(state:State):
            messages=[]
            llm = MCPNeo4J(self.neo4j_ai)
            e2e_Graph = self.graph_sys_prompt

            messages = state["messages"]
            agent = llm
            try:
                await agent.connect()
                data = await agent.chat(messages,e2e_Graph)
                # Ensure content is never None
                if data and hasattr(data, "text") and data.text is not None:
                    content = data.text
                elif data:
                    content = str(data)
                else:
                    content = "No response received from AI"
                
                return {"messages":[("assistant",content)],
                        "instructions":content,
                        "new_instructions":None #Basically setting it up so that this will be sent to the user, and if there is new instructions from the user in Tester it will be those new instructions, else there will be nothing
                        }
            
            except Exception as e:
                error_msg = f"Error in MCPGraph: {str(e)}"
                print(f"Fatal error during execution: {e}")
                traceback.print_exc()
                return {"messages": [("assistant", error_msg)],
                        "instructions": error_msg,
                        "new_instructions": None}
            
            finally:
                await llm.cleanup()

        async def MCPTester(state:State):
            messages=[]
            llm = MCPPlaywright(self.tester_ai,self.max_AI_steps,self.headless)
            e2e_Tester = self.tester_sys_prompt

            messages = state["messages"]

            new_instruct = state.get("new_instructions",None)


            agent = llm
            try:
                await agent.connect()
                if new_instruct:
                    data, tool_history = await agent.chat(new_instruct,e2e_Tester)
                else:
                    data, tool_history = await agent.chat(messages,e2e_Tester)
                
                # Ensure content is never None
                if data and hasattr(data, "text") and data.text is not None:
                    content = data.text
                elif data:
                    content = str(data)
                else:
                    content = "No response received from AI"
                
                console_cont = Console()
                console_cont.print("[magenta]-----------------------------[/magenta]")
                console_cont.print(Panel(f"[bold green] {content} [/bold green]", title="Final response"))

                test_type = self.test_type

                if test_type != "Parameter":
                    base_path = location(results=True, test_type=test_type)
                    os.makedirs(base_path, exist_ok=True)
                    filename = state.get("filename", "test_report.txt") 
                    full_path = os.path.join(base_path, filename)
                    with open(full_path, "w") as f:
                        f.write(content)

                return {"messages": [("assistant", content)], "tool_history": tool_history}
            
            except Exception as e:
                error_msg = f"Error in MCPTester: {str(e)}"
                print(f"Fatal error during execution: {e}")
                traceback.print_exc()
                return {"messages": [("assistant", error_msg)], "tool_history": []}
            
            finally:
                await llm.cleanup()

        async def generate_code(state: State): #generates the actual code from tool history
            tools = state.get("tool_history", [])
            tools_str = json.dumps(tools, indent=2)
            response_text = await generator(tools_str,self.code_generator_ai,self.generate_code_system_prompt)
            response = clean_code_block(response_text)
            
            if self.test_type == "Parameter" and self.csv_path:
                # Create the CSV reader wrapper
#                 csv_wrapper = f'''import pandas as pd
# import os
# import asyncio
# from playwright.async_api import async_playwright, Page, expect

# csv_path = r"{self.csv_path}"
# df = pd.read_csv(csv_path)

# print(f"Running {{len(df)}} test cases from CSV")

# async def main():
#     async with async_playwright() as playwright:
#         browser = await playwright.chromium.launch(headless={str(self.headless)})

#         for index, row in df.iterrows():
#             print(f"\\\\n=== Test Case {{index + 1}}/{{len(df)}} ===\")
#             print(f"Input values: {{dict(row)}}")
#             context = await browser.new_context()
#             page = await context.new_page()        
#             try:
# '''
                # Indent the AI-generated code (8 spaces for inside try block)
                # indented_response = "\n".join("            " + line if line.strip() else "" for line in response.split("\n"))
                indented_response = "\n".join(line if line.strip() else "" for line in response.split("\n"))
#         #         csv_footer = f'''
#         #         except Exception as e:
#         #             print(f"Test case {{index + 1}} FAILED: {{e}}")
#         #         else:
#         #             print(f"Test case {{index + 1}} PASSED")
#         #         finally:
#         #             context.close()
#         # browser.close()

# if __name__ == "__main__":
#     asyncio.run(main())
#     print("\\\\nAll tests completed!")
# '''
            # response = csv_wrapper + indented_response + csv_footer
            response = indented_response
            timestamp = datetime.datetime.now()
            unique_filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
            
            default=f"{self.test_type}_{unique_filename}.py"
            
            filename=state.get("filename",default)

            base_path = location(results=False,test_type=self.test_type)
            os.makedirs(base_path,exist_ok=True)
            full_path=os.path.join(base_path,filename)
            with open(full_path,"w") as f:
                f.write(response)
            return {"messages": [("assistant", f"Code generated and saved to {full_path}")]}

        graph = StateGraph(State)
        graph.add_node("Label_setup", LabelSetupNode)
        graph.add_node("Vector_search",VectorSearchNode)
        graph.add_node("MCPGraph", MCPGraph)
        graph.add_node("Embedding",EmbeddingNode)
        graph.add_node("Tester",MCPTester)
        graph.add_node("generate",generate_code)


        graph.add_edge("Label_setup", "Embedding")
        graph.add_edge("Embedding","Vector_search")
        graph.add_edge("Vector_search","MCPGraph")
        graph.add_edge("MCPGraph","Tester")
        graph.add_edge("Tester","generate")
        
        graph.set_finish_point("generate")
        embeddings_exist=check_embeddings()
        if embeddings_exist:
            graph.set_entry_point("Vector_search")
        if not embeddings_exist:
            graph.set_entry_point("Label_setup")

        graph_final = graph.compile(checkpointer=self.memory)
        # try:
        #     png_data = graph_final.get_graph().draw_mermaid_png()
        #     with open("graph.png", "wb") as f:
        #         f.write(png_data)
        #     print("Graph saved to graph.png")

        # except Exception as e:
        #     print(f"Error generating graph: {e}")
        return graph_final


if False:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
    async def run():
        async for event in graph_final.astream(
            {"messages": [("user", user_input)]}
            ):
                for value in event.values():
                    if value is not None and "messages" in value:
                        last_message = value["messages"][-1] 
                        if hasattr(last_message, 'content'):
                            # if an object
                            response = last_message.content
                        else:
                            #  a tuple ("user", "query")
                            response = last_message[1]
                        print("Assistant:", response)
                    else:
                        print("No output") # to handle NoneType

    asyncio.run(run())