from openai import OpenAI
import json
import tiktoken

# TODO: integrate with LangChain

def openai_evaluator(model='gpt-3.5-turbo', eval_prompt=None):

    if eval_prompt is None:
        raise ValueError("eval_prompt must be provided")

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": eval_prompt
            }
        ],
        response_format={'type': "json_object"}
    )
    resp = json.loads(completion.choices[0].message.content)
    return resp.get('score', None)

def openai_call(model='gpt-3.5-turbo', prompt=None):
    if prompt is None:
        raise ValueError("prompt must be provided")
    
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message.content

def token_encoder(decoded=None):
    if decoded is None:
        raise ValueError("prompt must be provided")
    
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(decoded)

def token_decoder(encoded=None):
    if encoded is None:
        raise ValueError("encoded must be provided")
    
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(encoded)