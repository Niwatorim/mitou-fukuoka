import json
import subprocess

def parse_code(code_string):
    values=subprocess.run(
        ["node","ast_generator.js"],
        input=code_string,
        capture_output=True,
        text=True,
        check=True,
    )
    return values.stdout

file_path = "./samples/App.jsx"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

ast_code=parse_code(content)

ast_data=json.loads(ast_code)
with open('astApp.json', 'w') as f:
    json.dump(ast_data, f, indent=4)

'''
loop the directory, looking for jsx and js files
make ast json for every page/file
each chunk is for each page
chunk = {
imports: import_code -> [] containing each import code line
main_component_function: just the name of the component
hook_usage: hook_code -> []
return: { under JSXElement
always_rendered: code -> find static components
conditional_rendering: code 
}
}
'''