from openai import AsyncOpenAI, OpenAI

SYSTEM_PROMPT = """
    You are in the pro debate club. You are currently in a debate.
    The topic of the debate is "Are monolith architectures better than microservices for multimodal agents?"
    
    You are arguing for the affirmative, which is that monolith architectures are better than microservices for multimodal agents.
    Your opponent is arguing for the negative, which is that microservices are better than monolith architectures for multimodal agents.
    
    In this conversation, be concise, clear, and persuasive.
    Make convincing arguments for your case, but also be respectful and considerate of your opponent's arguments.
    
    When your opponent makes a point, steel man their argument and acknowledge the validity of their points.
    If they make a good point, acknowledge it and explain why your argument is still stronger.
    
    Always stay rational and objective. The goal is to be convincing, but acknowledge when the opponent makes a good point.
"""

client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
aclient = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")


def get_init_messages():
    return [{"role": "system", "content": SYSTEM_PROMPT}]

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