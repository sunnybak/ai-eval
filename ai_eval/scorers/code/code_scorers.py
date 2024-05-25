import ast

def is_valid_python_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False




# Example usage
code_snippet = '''
def greet(name):
    prindt_(f"Hello, {name}!") : is None
'''

# code_snippet += ' is None'

if is_valid_python_syntax(code_snippet):
    print("The code snippet is valid.")
else:
    print("The code snippet has syntax errors.")