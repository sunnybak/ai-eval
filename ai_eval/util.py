from openai import OpenAI, AsyncOpenAI
import tiktoken
import numpy as np

def openai_call(prompt=None, model='gpt-3.5-turbo', response_format=None):
    if prompt is None:
        raise ValueError("prompt must be provided")
    
    client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=response_format
    )
    return completion.choices[0].message.content

async def a_openai_call_msg(messages=None, model='gpt-3.5-turbo'):
    if messages is None:
        raise ValueError("prompt must be provided")
    
    client = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    completion = await client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

async def a_openai_call(prompt=None, model='gpt-3.5-turbo', response_format=None):
    if prompt is None:
        raise ValueError("prompt must be provided")
    
    client = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=response_format
    )
    return completion.choices[0].message.content

def cosine_similarity(string_a, string_b, model='text-embedding-3-small'):
    client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    vector_a = client.embeddings.create(input = [string_a], model=model).data[0].embedding
    vector_b = client.embeddings.create(input = [string_b], model=model).data[0].embedding
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_str_token_list(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_integer = encoding.encode(text)
    tokens_bytes = [encoding.decode_single_token_bytes(token) for token in tokens_integer]
    str_list = [b.decode('utf-8').strip() for b in tokens_bytes]
    return str_list
