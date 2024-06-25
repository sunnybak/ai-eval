from typing import Callable
from ai_eval.evaluators import BaseEvaluator
from groq import AsyncGroq
from ai_eval.generator.chat_gen import run
from ai_eval.targets import ContainsTarget
import json

client = AsyncGroq(api_key="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")
async def a_topic_scorer(topic, messages):
    response = await client.chat.completions.create(
        messages=messages + [
            {'role': 'system', 'content': 'The conversation is now over. Answer questions about the conversation.'},
            {"role": "user", 
            "content": "Give a score between 1-10 on how closely the topic of the given conversation matches this topic: " + topic + 'Try giving scores that are closer to 1 and 10. Respond in JSON format with the following format: {"score": score}'}],
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


app = 'be a good legal assistant'
topic_eval = TopicScoreEvaluator('tax credits')
user = 'small business owner looking for tax advice'

run(chatbot=app, evaluator=topic_eval, user=user)
