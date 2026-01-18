from typing import Annotated,Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
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
import textwrap

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
    def __init__(self,test_type:str,neo4j_url:str,neo4j_database:str,neo4j_pwd:str,neo4j_ai_model:str,app_address:str,max_AI_steps:int,headless:bool,tester_ai:str,code_generator_ai:str,similarity_k:int=20,columns:list[str]=[],expected_columns:list[str]=[],csv_path:str=None):
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

        :param expected_columns: The columns of the dataframe being passed in
        :type expected_columns: list[str]
        
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
        self.expected_results = expected_columns
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
            expected_display = self.expected_results
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

                RESULT ELEMENT DISCOVERY (CRITICAL FOR CODE GENERATION):
                - After submitting the data, OBSERVE where the result/feedback message appears
                - Record the EXACT selector of the result element (e.g., "#login-result", ".success-message", "[data-testid='result']")
                - Note whether the result replaces content, appears as a new element, or redirects to a new page
                - This is ESSENTIAL for generating working test code
                - DO NOT RETURN ANY REFERENCE ID (e.g. ref = )
                
                OUTPUT FORMAT (STRICT):
                Field Mappings (use EXACT selectors like page.locator("#reg-email") or page.locator("#field-id")):
                {chr(10).join(f"- {col}: <exact_selector_with_id>" for col in columns) if columns else "- field1: <exact_selector_with_id>"}
                
                Example (if there is a submit button etc.)
                Submit Button: <exact_selector>
                
                Result Element: <exact_selector_where_result_appears>
                Result Sample: <actual_text_shown_after_submission>
            
            """ 
            self.graph_sys_prompt=f"""
                You are a graph-based form testing expert specializing in parameter validation.

                TASK: Find the form/component that accepts these inputs: {", ".join(columns) if columns else "None"}

                RULES:
                - Use tools to inspect the Neo4j graph
                - Find form components, input fields, and submit buttons
                - Match field names/IDs to the input columns: {", ".join(columns) if columns else "None"}
                - Identify where results/responses appear after submission
                - Be as specific as possible and give the unique selectors for each field, but do not return IDs that are from neo4j, but that are for the website
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
                target: .... MAKE SURE TO USE THE most unique selector, such as classname, id, etc.
                expect:....

                - step: 3
                action: submit
                instructions:...
                target: ...
                expect:....

                """
            #Might need a feedback loop or a question back to the mcp or something cuz this is oof
            self.generate_code_system_prompt=f"""
            You are a Senior QA Automation Engineer generating PARAMETERIZED test code.

            CONTEXT:
            - CSV Path: {self.csv_path}
            - Input Columns (EXACT NAMES FROM CSV): {column_display}
            - Expected Result Columns (EXACT NAMES FROM CSV): {expected_display}

            SELECTOR EXTRACTION (CRITICAL):
            - Look at the execution history/tool_history for the EXACT selectors that were used
            - Find where the result/feedback message appeared after form submission
            - Extract the result element selector (e.g., "#login-result", ".result-message")
            - Use ONLY selectors that were confirmed to work in the execution history

            COLUMN NAME RULES (CRITICAL - MUST FOLLOW EXACTLY):
            - Use ONLY the exact column names from the CSV: {column_display}
            - For expected results, use ONLY these column names: {expected_display}
            - DO NOT invent column names like "expected_response_email" unless they exist in the CSV
            - Example: If CSV has "expected_response_sucess", use row["expected_response_sucess"] NOT row["expected_response_email"]

            ASSERTION STRATEGY (USE FLEXIBLE MATCHING):
            - Use "contains" matching instead of exact matching for robustness
            - Example: assert expected_value.lower() in actual_result.lower(), f"Expected '{{expected_value}}' to be in '{{actual_result}}'"
            - This works across different apps that may have varying message formats
            - Handle None/empty values gracefully with str() conversion

            CRITICAL REQUIREMENTS:
            1. Access CSV data using EXACT column names: row["column_name"]
            2. For expected results, use the EXACT column name from CSV (e.g., row["{expected_display[0] if expected_display else 'expected_result'}"])
            3. For EACH input column, you must:
               - Get the value: value = str(row["column_name"]) if pd.notna(row["column_name"]) else ""
               - Fill the corresponding field using the selector from execution history
               - Example: await page.fill("#email-input", value)

            4. After filling all fields:
               - Click the submit button (use selector from execution history)
               - Wait for response: await page.wait_for_load_state("networkidle")
               - Extract actual result from the result element discovered during testing

            5. FLEXIBLE ASSERTION:
               - Get expected: expected = str(row["{expected_display[0] if expected_display else 'expected_result'}"])
               - Get actual: Extract text from result element
               - Assert with contains: assert expected.lower() in actual.lower() or actual.lower() in expected.lower()

            6. DO NOT include: imports, CSV reading loop, or main function
            7. Output ONLY the indented test logic (inside the try block)
            8. Use proper async/await syntax
            9. Add meaningful wait statements after actions that trigger loading
            10. Handle empty/None CSV values with: str(value) if pd.notna(value) else ""
            11. DO NOT USE REF VALUES THAT YOU HAVE BEEN PROVIDED, ONLY SELECTORS.

            BUT THE ONLY CODE YOU WILL ADD IS BETWEEN THE TRY BLOCK
            ```python
            
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
                    await browser.close()
            ```

            OUTPUT REQUIREMENTS:
            - Must be valid Python with proper indentation
            - Must use async/await for all Playwright calls  
            - Must use EXACT column names from CSV (not invented ones)
            - Must use flexible "contains" assertions
            - Must handle empty values gracefully
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

            # llm UI extractor
            llm = ChatOllama(
            model="qwen2.5:1.5b",
            temperature=0 
            )

            prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a web QA tester. Extract the UI components and actions from the prompt, and put them as a list. For example, Prompt: Check whether the home button has the home logo, and directs to the shop link, and whether the cat image is present. Response you should give: home button, home logo, shop link, cat image. ONLY USE A COMMA AS THE SEPARATOR"),
            ("user", "{question}")
            ])

            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({"question": user_query})
            print(response)
            prompt_array = [item.strip() for item in response.split(',')]
            print(prompt_array)

            seen_ids = set()
            all_test_reports = [] # To store the formatted text for each found component

            def format_section(title, items):
                section = [f"\n--{title}--"]
                if not items:
                    section.append("None found")
                else:
                    for i, item in enumerate(items, 1):
                        raw_code = item.get('code', '')
                        clean_code = re.sub(r'\s+', ' ', raw_code).strip()
                        display_code = clean_code[:100] + "..." if len(clean_code) > 100 else clean_code
                        section.append(f"{i}. ID: {item.get('id')} | Code: {display_code}")
                return "\n".join(section)

            for search_term in prompt_array:
                results = store.similarity_search_with_score(search_term, k=1)
                if not results:
                    continue

                document, score = results[0]
                if score < 0.80:
                    continue
                
                node_id = document.metadata.get('id')
                if node_id in seen_ids:
                    continue
                
                seen_ids.add(node_id)
                meta = document.metadata

                # Build the report for THIS specific component
                comp_output = []
                comp_output.append(f"Found: {meta.get('name')} (ID: {node_id}) Score: {score:.4f}")
                comp_output.append(format_section("LINKS", meta.get('links', [])))
                comp_output.append(format_section("IMAGES", meta.get('images', [])))
                comp_output.append(format_section("INPUTS", meta.get('inputs', [])))
                comp_output.append(format_section("BUTTONS", meta.get('buttons', [])))
                comp_output.append(format_section("ROUTES", meta.get('routes', [])))
                
                # Add this individual component report to our collection
                all_test_reports.append("\n".join(comp_output))

            # 5. Final Response
            if not all_test_reports:
                text = "No components found matching the criteria."
                return {"messages": [("assistant", text)]}

            final_response = "\n\n" + "="*30 + "\n"
            final_response += "\n\n".join(all_test_reports)
            
            print("--- Final Aggregated Results ---")
            print(final_response)
            return {"messages": [("assistant", final_response)]}
            # seen_ids = set()
            # unique_components = []

            # print(f"--- Searching for {len(prompt_array)} items: {prompt_array} ---")
            # for search_term in prompt_array:
            #     print(f"Searching for: '{search_term}'")

            #     k=self.similarty_k
            #     results = store.similarity_search_with_score(search_term, k=k) 

            #     for document, score in results:
            #         if score < 0.70: 
            #             continue
                    
            #         node_id = document.metadata.get('id')
            #         if node_id in seen_ids:
            #             continue
                    
            #         seen_ids.add(node_id)
                    
            #         meta = document.metadata
            #         item_str = f"Name: {meta.get('name', 'Unnamed')} | ID: {node_id} | Code: {meta.get('code')} | Score: {score:.4f}"
            #         unique_components.append(item_str)

            # if not unique_components:
            #     text = "No components found matching the criteria."
            #     print(text)
            #     return {"messages": [("assistant", text)]}

            # final_response = "\n".join(unique_components)
            # print("--- Final Aggregated Results ---")
            # print(final_response)
            # return {"messages": [("assistant",final_response)]}

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
            chat=state.get("messages",[])
            last_msg=chat[-1]
            tools_str = json.dumps(tools, indent=2)
            response_text = await generator(tools_str,last_msg,self.code_generator_ai,self.generate_code_system_prompt)
            response = clean_code_block(response_text)
            
            if self.test_type == "Parameter" and self.csv_path:
                # Create the CSV reader wrapper
                csv_wrapper = f'''import pandas as pd
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
'''
                #TODO: Test E2E with multiple pages and make demo
                # Fix indentation: AI often returns first line unindented but rest indented
                lines = response.splitlines()
                if lines:
                    first_line = lines[0].strip()  # First line, stripped
                    if len(lines) > 1:
                        # Dedent remaining lines to remove their excess indentation
                        remaining = textwrap.dedent("\n".join(lines[1:]))
                        remaining_lines = remaining.splitlines()
                        # Build final response with consistent 16-space indent
                        indented_lines = ["                " + first_line]
                        for line in remaining_lines:
                            indented_lines.append("                " + line if line.strip() else "")
                        indented_response = "\n".join(indented_lines)
                    else:
                        indented_response = "                " + first_line
                else:
                    indented_response = ""

                csv_footer = textwrap.dedent(f'''
            except Exception as e:
                print(f"Test case {{index + 1}} FAILED: {{e}}")
            else:
                print(f"Test case {{index + 1}} PASSED")
            finally:
                await context.close()
        await browser.close()             

if __name__ == "__main__":
    asyncio.run(main())
    print("\\nAll tests completed!")
''')
                # response = csv_wrapper + indented_response + csv_footer
                response = csv_wrapper + "\n" + indented_response + csv_footer

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