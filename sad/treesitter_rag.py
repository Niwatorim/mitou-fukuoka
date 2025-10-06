import os
from tree_sitter_languages import get_parser
from langchain_core.documents import Document

def tree_sitter_chunks(file_path:str, language:str) -> list[Document]:
    parser = get_parser(language)

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    tree = parser.parse(bytes(code, 'utf8'))
    return tree
    # chunk_nodes = [
    #     'function_declaration',
    #     'class_declaration',
    #     'method_definition',
    #     'arrow_function',
    #     'lexical_declaration'
    # ]
    # chunks = []

    # def transverse_tree(node): # visit each node in the tree once
    #     if node.type in chunk_nodes:
    #         start_line = node.start_point[0] + 1
    #         end_line = node.end_point[0] + 1

    #         chunk_content = node.text.decode('utf-8')

    #         chunk_doc = Document(
    #             page_content= chunk_content,
    #             metadata={
    #                 ""
    #             }
    #         )

def visualize_tree(node, indent=""):
    """Recursively prints the nodes of the tree with their type and position."""
    
    start_line, start_col = node.start_point
    end_line, end_col = node.end_point
    
    print(
        f"{indent}{node.type}  "
        f"[(L{start_line + 1}, C{start_col}) - (L{end_line + 1}, C{end_col})]"
    )
    
    # Recurse into children
    for child in node.children:
        visualize_tree(child, indent + "  ")

file_path = "./reactapp/my-react-app/src/App.jsx"
tree = tree_sitter_chunks(file_path, 'tsx')

print("--- Tree Visualization ---")
visualize_tree(tree.root_node)

