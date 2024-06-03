

# in this file, we will create a chat simulator
# which takes a user prompt, assistant prompt, and
# the exit condition for the user
# we will use Groq + Llama3 to run the simulation async
# we will pretty print the chat messages to visualize
# we will also report the number of tokens and cost of the simulation
# we will run the simulation conversation through a chat scorer
# to get a full breakdown of the chat
from pydantic import BaseModel
import asyncio
from groq import AsyncGroq
from ai_eval.scorers.chat.chat_scorer import ChatScorer
from openai import AsyncOpenAI
import json
client = AsyncGroq(
    api_key='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',
)

oai_client = AsyncOpenAI(
    api_key='sk-proj-2UDNk6vwtcfFuabLNGVFT3BlbkFJor4tAlnwexn6RgCFKoBc',
)

class Chat(BaseModel):
    messages: list[dict[str, str]]
    
    # implement messages validator
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_chat(messages):
    for m in messages:
        if m["role"] == 'user':
            print(bcolors.OKBLUE + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)
        elif m["role"] == 'assistant':
            print(bcolors.OKGREEN + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)
        elif m["role"] == 'system':
            print(bcolors.HEADER + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)

def get_sys_prompt_from_chat(messages: Chat) -> str: 
    pass
    

async def simulate_chat(app_prompt: str, 
                        user_prompt: str = None, 
                        stop_word: str = 'END',
                        max_tokens: int = 10000,
                        max_messages: int = 10,
                        max_cost: float = 1, # $
                        max_time: float = 10, # in seconds
                        ) -> Chat:
    # the simulated chat starts with the assistant prompt
    # the first message is from the user
    # the second message is from the assistant
    
    if user_prompt is None:
        gen_user_prompt = '''
            Your task is to create an LLM prompt for a synthetic user who will use an LLM App to test it out.
            The user will provide the app's objective below.
            Based on this, think step by step about the profile, needs, and background of the user.
            Then, generate an LLM system prompt for a synthetic user to use the App to test it out.
            Start the GENERATED_USER_PROMPT like - You are an intelligent, calm, direct human. Your task is to use an LLM App for the following objective: 
            Also make sure the synthetic user to keep messages short and conversational.
            Respond with a json object that has the following format: {"role": "user", "content": GENERATED_USER_PROMPT}
        '''
        
        response = await oai_client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': gen_user_prompt},
                {'role': 'user', 'content': "Here is the objective of the App: " + app_prompt},
            ],
            model="gpt-4o",
            response_format={ 'type': "json_object" },
        )
        user_prompt = json.loads(response.choices[0].message.content)['content']
        print(bcolors.FAIL + 'SYNTH USER PROMPT: \n\n', user_prompt, bcolors.ENDC)
    
    messages_app = [{'role': 'system', 'content': app_prompt}]
    messages_user = [{'role': 'system', 'content': user_prompt}]
    total_cost = 0
    total_tokens = 0
    turn_user = True
    
    start_clock = asyncio.get_event_loop().time()
    while \
        max(len(messages_app), len(messages_user)) < max_messages and \
        (duration := asyncio.get_event_loop().time() - start_clock) < max_time and \
        total_cost < max_cost and \
        total_tokens < max_tokens:

        # get the turn's messages
        messages = messages_user if turn_user else messages_app

        # get the new message
        stream = await client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            stream=True,
        )
        new_message = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                new_message += content

        if turn_user:
            messages_user.append({"role": 'assistant', "content": new_message})
            messages_app.append({"role": 'user', "content": new_message})
        else:
            messages_user.append({"role": 'user', "content": new_message})
            messages_app.append({"role": 'assistant', "content": new_message})
        
        if stop_word in new_message:
            break
        
        turn_user = not turn_user
    
    # stats
    chat_scorer = ChatScorer(messages_app)
    counts = chat_scorer.chat_total_tokens()
    cost = 0.05e-6*counts['input'] + 0.08e-6*counts['output']
    total_tokens = counts['input'] + counts['output']
    avg_tps = total_tokens / duration
    
    # report
    print_chat(messages_app)
    print(f"Total cost: {cost}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average TPS: {avg_tps}")
    print(f"Elapsed time: {duration}")
    
    return Chat(messages=messages_app)


def run(chatbot):
    # simulate_chat(
    #     user_prompt='Talk as an intelligent and direct human who is using an LLM assistant. \
    #         Your task is to use the assistant to do paper napkin math to evaluate the impact of LLMs on software. \
    #         Once you have the answer, say the capitalized word "END" to end the chat. \
    #         Keep responses short and conversational',
    #     app_prompt='Help the user. Keep responses very short and conversational. Think step by step.',
    #     user_exit_condition='exit',
    #     max_tokens=100000,
    #     max_messages=10,
    #     max_time=15,
    #     max_cost=1000,
    # )
    asyncio.run(simulate_chat(
        app_prompt=chatbot,
    )
        
    )    




chatbot = "help the user generate a observability software PRD based off of an idea"
synth_user = "you are a student"
format_eval = 'pdf'
run(chatbot, synth_user, format_eval(guardrail=True))


    

