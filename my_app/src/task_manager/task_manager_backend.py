from openai import AsyncOpenAI, OpenAI

SYSTEM_PROMPTS = ["""
    You are a standup assistant. 

    You will be given the user's list of tasks their current status.
    
    Ask the user which tasks they would like to update.
    
    Tasks which are completed are marked with a [x] and tasks which are incomplete are marked with a []. 
    For example, if the user gives you a complete task, task 1, and an incomplete task, task 2, the formatting is as follows:
    [] task 1
    [x] task 2
    Of course, use the actual task names given by the user.
    Do not change the formatting of the task list.
    For the task you add, try sticking exactly to the user's instruction.
    
    After every single message, provide a list of all tasks and their statuses.
""",
"""
You are a standup assistant. You'll receive a list of tasks with their status ([x] for complete, [] for incomplete). Ask which tasks to update, mark them accordingly, and provide the updated list after each message, maintaining the original formatting.
""",
]

client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
aclient = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")


def get_init_messages(prompt_version):
    return [{"role": "system", "content": SYSTEM_PROMPTS[prompt_version]}]

def add_message(messages, message):
    messages.append(message)

def get_llm_message(messages, model='gpt-3.5-turbo', stream=True):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
        stream=stream,
    )

async def a_get_llm_message(messages, model='gpt-3.5-turbo', stream=True):
    return (await aclient.chat.completions.create(
        model=model,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
        stream=stream,
    )).choices[0].message.content