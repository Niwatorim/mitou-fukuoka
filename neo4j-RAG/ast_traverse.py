import json
import subprocess
import os

root_dir = "../samples"
ast_dir = "../samples_ast"

# to parse each file into AST json
def ast_parser(code_string):
    values=subprocess.run(
        ["node","ast_generator.js"],
        input=code_string,
        capture_output=True,
        text=True,
        check=True,
    )
    return values.stdout

# to take the testable attributes
def ast_rag(file):
    parser_path= "./parser_test.js"
    command = ["node",parser_path,file]
    values=subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return values.stdout

# --- The Traversal Logic ---

def get_node_text(node, source_code):
    """Slices the source code to get the text of a specific AST node."""
    return source_code[node['start']:node['end']]

def get_jsx_element_name(node):
    """Safely gets the name of a JSX element (e.g., 'Sidebar')."""
    if node and node.get('type') == 'JSXElement':
        return node.get('openingElement', {}).get('name', {}).get('name')
    return None

def _find_attribute_value(jsx_element_node, attribute_name):
    """Finds the value of a specific attribute in a JSX element's attributes list."""
    if 'attributes' in jsx_element_node.get('openingElement', {}):
        for attr in jsx_element_node['openingElement']['attributes']:
            if attr.get('name', {}).get('name') == attribute_name:
                # Handle simple string values like path="/login"
                if attr.get('value', {}).get('type') == 'StringLiteral':
                    return attr['value']['value']
    return None

# Helper function to analyze the outcome of a conditional (e.g., <HomePage /> or <Navigate...>)
def _analyze_outcome_node(outcome_node, source_code):
    """Analyzes a node to determine if it's a Component render or a Navigation action."""
    if outcome_node.get('type') == 'JSXElement':
        component_name = outcome_node.get('openingElement', {}).get('name', {}).get('name')
        if component_name == 'Navigate':
            # It's a navigation action, find the 'to' prop
            target_path = _find_attribute_value(outcome_node, 'to')
            return {"type": "Navigation", "to": target_path}
        else:
            # It's a regular component render
            return {"type": "Component", "name": component_name}
    # Fallback for other potential node types
    return {"type": "Unknown", "text": get_node_text(outcome_node, source_code)}

def traverse(node, results, source_code):
    """
    Recursively traverses the AST, collecting information into the results dictionary.
    This is the core of the solution.
    """
    if not isinstance(node, dict):
        return

    # 1. Find Imports and Hooks
    if node.get('type') == 'ImportDeclaration':
        import_text = get_node_text(node, source_code)
        results['imports'].append(import_text)
        # Check specifiers for hooks
        for specifier in node.get('specifiers', []):
            imported_name = specifier.get('local', {}).get('name', '')
            if imported_name.startswith('use'):
                results['hook_usage'].append(import_text)

    # 2. Find the Main Component Function
    # Handles `const MyComponent = () => {}` and `function MyComponent() {}`
    if node.get('type') == 'VariableDeclaration':
        for declaration in node.get('declarations', []):
            if declaration.get('init', {}).get('type') == 'ArrowFunctionExpression':
                component_name = declaration.get('id', {}).get('name')
                if component_name and component_name[0].isupper(): # Convention for components
                    results['main_component_function'] = component_name
                    # Now traverse inside this component's body
                    traverse(declaration.get('init', {}).get('body'), results, source_code)

    if node.get('type') == 'FunctionDeclaration':
        component_name = node.get('id', {}).get('name')
        if component_name and component_name[0].isupper():
            results['main_component_function'] = component_name
            # Traverse inside this component's body
            traverse(node.get('body'), results, source_code)


    # 3. Find Return Statement and analyze JSX
    if node.get('type') == 'ReturnStatement':
        # The returned content is in the 'argument'
        traverse(node.get('argument'), results, source_code)

    # 4. find conditional rendering elements and ROUTES
    if node.get('type') == 'JSXElement' and node.get('openingElement', {}).get('name', {}).get('name') == 'Route':
        route_info = {
            "path": None,
            "element": {
                "isConditional": False,
                "render": None # For non-conditional routes
            }
        }

        attributes = node.get('openingElement', {}).get('attributes', [])
        for attr in attributes:
            attr_name = attr.get('name', {}).get('name')
            attr_value_node = attr.get('value', {})

            if attr_name == 'path' and attr_value_node.get('type') == 'StringLiteral':
                route_info['path'] = attr_value_node.get('value')

            if attr_name == 'element':
                # Case 1: Non-conditional element like element={<SettingsPage />}
                if attr_value_node.get('type') == 'JSXElement':
                    route_info['element']['render'] = _analyze_outcome_node(attr_value_node, source_code)

                # Case 2: Conditional element like element={authUser ? ... : ...}
                elif attr_value_node.get('type') == 'JSXExpressionContainer':
                    expression = attr_value_node.get('expression', {})
                    if expression.get('type') == 'ConditionalExpression':
                        route_info['element']['isConditional'] = True
                        route_info['element']['condition'] = get_node_text(expression.get('test'), source_code)
                        route_info['element']['true_outcome'] = _analyze_outcome_node(expression.get('consequent'), source_code)
                        route_info['element']['false_outcome'] = _analyze_outcome_node(expression.get('alternate'), source_code)
                        # Clear the non-conditional render key
                        del route_info['element']['render']

        results['routes'].append(route_info)

    # --- The Recursive Step ---
    # Continue traversing through all possible children of the current node
    for key, value in node.items():
        if isinstance(value, dict):
            traverse(value, results, source_code)
        elif isinstance(value, list):
            for item in value:
                traverse(item, results, source_code)

# hybrid method
# def create_rag_chain(): 
#     traversal_config = Eager( # need to experiemtn with this, how deep the graph is traversed
#         select_k=5,
#         start_k=2,
#         adjacent_k=3,
#         max_depth=2
#     )

#     graph_retriever = GraphRetriever(
#         store=vector_store,
#         strategy=traversal_config
#     )

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
    
#     prompt = ChatPromptTemplate.from_template("""
# # prompt from ast here
# """)
    
#     chain = (
#         {
#             "graph_context": graph_retriever | format_docs,
#             "question": RunnablePassthrough()
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain
