import ast
from ai_eval.evaluators.scorers.AbsBaseScorer import AbsBaseScorer

class CodeScorer(AbsBaseScorer):
    def is_valid_python_syntax(code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False