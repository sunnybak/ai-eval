from ai_eval import ChatAppSettings, ChatEvals, ChatConsole, SynthChatUsers
from ai_eval.chat import Reason, MaxBackAndForth
from ai_eval.evals import MaxCostEval, MaxTokenEval, MaxTokensEval, ToneEval, EmotionEval, SentimentEval



messages = ai.save(messages)











# AI Settings of the app
app = ChatAppSettings(
    app_prompt = 'Help the user with their medical condition', 
    model_settings = {'model': 'gpt-3.5-turbo'},
    api_key = '',
    tokenizer = '',
)

# Evals of the app
evals = ChatEvals(
    ai.eval(point='chat_messages') := [SentimentEval(), EmotionEval(), MaxCostEval(1, assert_eval=True)],
    ai.eval(point='single_message') := [ToneEval(), MaxTokensEval(1), MaxTokenEval(1, assert_eval=True)],
)

# chat.complete(messages=messages, model='gpt-3.5-turbo', stream=True)
# post_processed_messages = messages + extra_message
# result = ai.eval(point='chat_messages', data=post_processed_messages)

# Tests of the app
tests = SynthChatUsers(
    profiles = ['grandma in New York', 'college student in India'],
    needs = ['back ache', 'headache'],
    paths = ['cure', 'emotional support'],
    styles = ['casual slang', 'lot of typos'],
    languages = ['en', 'es'],
    stop = [Reason('problem solved'), MaxBackAndForth(5)],
)

console = ChatConsole(app, evals, tests)
console.run() # runs until the stopping point


# profile, needs, paths, styles, stop
# seed(s)
# empty list autoseeds based on ProductInverse(product objective)