from openai import OpenAI

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

openai_call('hi')
