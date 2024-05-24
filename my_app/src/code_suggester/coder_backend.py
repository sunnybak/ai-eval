from openai import AsyncOpenAI, OpenAI

SYSTEM_PROMPTS = [
    """
    You are a coding assistant. The user will instruct you to write Python code snippets.
    Make sure you follow the user's instructions explicitly and provide the correct Python code.
    The Python code MUST be executable.
    """,
    """
    You are a coding assistant.
    """,
]

client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
aclient = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")


def get_init_messages(sys_prompt_version=0):
    return [{"role": "system", "content": SYSTEM_PROMPTS[sys_prompt_version]}]

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