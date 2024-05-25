from openai import AsyncOpenAI, OpenAI

SYSTEM_PROMPTS = ["""
    Chat with the user about quantum entanglement and the beginning of time. 
    Keep responses very short.
""",
]

client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
aclient = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")


def get_init_messages():
    return [{"role": "system", "content": SYSTEM_PROMPTS[0]}]

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