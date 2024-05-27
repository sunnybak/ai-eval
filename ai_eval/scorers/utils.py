from openai import OpenAI
import requests

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

def calculate_cost(provider, model, prompt_tokens, completion_tokens):
    # https://openrouter.ai/api/v1/models
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    data = response.json()
    model_list = data['data']

    found_model = None
    provider = provider.lower()
    model_id = provider + "/" + model
    for model in model_list:
        if model['id'] == model_id:
            found_model = model
            break
    if found_model is None:
        raise ValueError(f"Model {model_id} not found in openrouter list")

    pricing = found_model['pricing']
    cost = (completion_tokens * float(pricing['completion'])) + (prompt_tokens * float(pricing['prompt']))
    return cost

