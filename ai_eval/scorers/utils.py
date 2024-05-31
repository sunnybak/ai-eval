from openai import OpenAI
import os
import google.generativeai as genai
import requests
def gemini_call(prompt=None, model='gemini-1.5-pro-latest'):
    if prompt is None:
        raise ValueError("prompt must be provided")
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    gemini_model = genai.GenerativeModel(model, generation_config=generation_config)
    chat = gemini_model.start_chat()
    result = chat.send_message(prompt)
    response = result.text
    return response

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

