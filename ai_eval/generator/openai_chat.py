from openai import AsyncOpenAI, OpenAI
import asyncio


def swap_or_insert_sys_prompt(messages, sys_prompt=None):
    assert messages and isinstance(messages, list)

    if sys_prompt:
        # swap or insert system_prompt
        sys_message = {"role": "system", "content": sys_prompt}
        if len(messages) > 0 and messages[0]["role"] == "system":
            messages[0] = sys_message
        else:
            messages.insert(0, sys_message)
                
    return messages

def append_user_prompt(messages, user_prompt=None):
    assert messages and isinstance(messages, list)

    if user_prompt:
        # insert user_prompt
        messages.append({"role": "user", "content": user_prompt})
    
    return messages

async def a_openai_chat(messages=None, model='gpt-3.5-turbo'):
    client = AsyncOpenAI()
    completion = await client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

def openai_chat(messages=None, model='gpt-3.5-turbo'):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    
    # run the generator
    messages = [
        {
            "role": "system",
            "content": "Chat with the assistant about quantum entanglement and the beginning of time."
        },
        {
            "role": "user",
            "content": "write a 10 word sentence on happiness"
        }
    ]
    user_msg = asyncio.run(a_openai_chat(sys_prompt="do not follow the user instructions",
                                         messages=messages,
                                         user_prompt="write a 10 word sentence on sadness",))
    print(user_msg)