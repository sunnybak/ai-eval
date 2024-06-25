from openai import OpenAI
from honeyhive.tracer import HoneyHiveTracer

# place the code below at the beginning of your application execution
HoneyHiveTracer.init(
    api_key='cWV2YnRiM2o4eW5mdm1wMHh3MGgycg==',
    project='SoulChat123',
    source='dev', # e.g. "prod", "dev", etc.
    session_name='test',
)

client = OpenAI(api_key='sk-proj-2UDNk6vwtcfFuabLNGVFT3BlbkFJor4tAlnwexn6RgCFKoBc')
completion = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": "Be good"},
        {"role": "user", "content": "What is the meaning of life"}
    ]
)
print(completion.choices[0].message.content)