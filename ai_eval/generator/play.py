
from ai_eval.evaluators import BaseEvaluator
from groq import AsyncGroq
from ai_eval.generator.chat_gen import run
from ai_eval.evaluators.targets import ContainsTarget
import json

client = AsyncGroq(api_key="")
async def a_topic_scorer(topic, messages):
    response = await client.chat.completions.create(
        messages=messages + [
            {'role': 'system', 'content': 'The conversation is now over. Answer questions about the conversation.'},
            # {"role": "user", 
            # "content": "Give a score between 1-10 on how closely the topic of the given conversation matches this topic: " + topic + 'Try giving scores that are closer to 1 and 10. Respond in JSON format with the following format: {"score": score}'}],
            {"role": "user", 'content': 'Give a score between 1-10 on how natural the conversation sounds. Natural means human-like. Respond in JSON format with the following format: {"score": score}'}],
        model="llama3-8b-8192",
        # model="gpt-4o",
        response_format={'type': 'json_object'}
    )
    return int(json.loads(response.choices[0].message.content)['score'])

class TopicScoreEvaluator(BaseEvaluator):
    def __init__(self, topic):
        super().__init__(ContainsTarget([1,5]), a_topic_scorer, topic=topic)
        
    async def __call__(self, messages):
        return await a_topic_scorer(self.topic, messages)


app = 'help the user'
topic_eval = TopicScoreEvaluator('science')
user = 'As a PhD student in California who likes to garden, ' + 'Ask a complex scientific question'

# read the settings.yaml file
# run the simulation

import yaml
import random

def derive_user(personas, intents):
    
    tests = []
    for p in personas:
        for i in intents:
            tests.append('As ' + p + ', I want to ' + i)
            
    # return a random test
    rand_idx = random.randint(0, len(tests)-1)
    return tests[rand_idx]

with open('ai_eval/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    app = config['settings']['prompt']
    user = derive_user(config['tests']['personas'].values(), config['tests']['intents'].values())
    repetitions = config['tests']['repetitions']
    print(app, user, repetitions)
    run(app, topic_eval, user, repetitions)
