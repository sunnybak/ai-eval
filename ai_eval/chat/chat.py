from ai_eval.chat.chat_backend import ChatBackend
from ai_eval.util import a_openai_call_msg
from ai_eval import Target, scorer, run_experiment


async def generator_add_gen(messages, back_and_forth=10):
    SYNTH_USER_PROMPT = """
        Chat with the assistant about quantum entanglement and the beginning of time. 
        Maintain conversational format and keep responses short.
    """
    model = 'gpt-3.5-turbo'
    
    # replace the system prompt with the synthetic user prompt
    messages[0]['content'] = SYNTH_USER_PROMPT

    # back and forths
    if (len(messages) - 1) // 2 < back_and_forth:
        user_msg = await a_openai_call_msg(
                            messages=messages, 
                            model=model)
        return user_msg
    return None

async def run_app(_, model, back_and_forth=10):
    
    backend = ChatBackend(model=model)

    # initialize messages
    messages = backend.get_init_messages()

    # chat simulation 
    user_msg = await generator_add_gen(messages, back_and_forth)
    print('\n\nUser: ', user_msg)
    while user_msg is not None:
        backend.add_message(messages, {"role": "user", "content": user_msg})
        llm_response = await backend.a_get_llm_message(messages, model=model, stream=False)
        print('\n\nAssistant: ', llm_response)

        backend.add_message(messages, {"role": "assistant", "content": llm_response})
        user_msg = await generator_add_gen(messages, back_and_forth)
        if user_msg is None: break
        print('\n\nUser: ', user_msg)

    # score the trajectory
    return []

def test_dummy():
    
    # simulate multi-turn chat
    # simulate agents
    
    score_df = run_experiment(
        app=run_app,
        args=(None,),
        hyperparam_dict={
            'model': ['gpt-3.5-turbo'],
        },
        consistency=1
    )
    
    # score_df['result'] = score_df['score'].apply(evaluator)
    # print(score_df[['duration']])
    # print(score_df[['scorer_args', 'scorer_kwargs']])
    # print(score_df[['model', 'score', 'result']])

