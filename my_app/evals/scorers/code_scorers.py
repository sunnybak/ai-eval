from ai_eval import scorer

# score is 1 if code is executable and 0 otherwise
# use python exec function
@scorer
def score_code_assistant(messages):
    # code is content of last msg
    code = messages[-1]["content"]
    try:
        code = code.split("```python")[1].split("```")[0]
        if code is None:
            return 0
        a = exec(code)
        return 1
    except:
        return 0

