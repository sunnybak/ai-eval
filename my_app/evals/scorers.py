from ai_eval.scorers.callable_scorer import CallableScorer
from ai_eval.util import openai_call


class SentiScorer(CallableScorer):
    def score(self, points):
        response = openai_call('gpt-3.5-turbo', 
            'Determine whether the sentiment of this message is cheerful or serious. \
            Respond only with the word "cheerful" or "serious". Here is the point: ' \
                + points[0])
        return response
    
class CompareScorer(CallableScorer):
    def score(self, points):
        print('Comparing points: ', points)
        return points[0] != points[1]
    
class ChihuahuaScorer(CallableScorer):
    def score(self, points):
        response = openai_call('gpt-3.5-turbo', 
            'Determine whether the image is of a chihuahua or a blueberry muffin or a tomato. \
            Respond only with the word "chihuahua" or "blueberry muffin". Here is the point: ' \
                + points[0])
        return response.lower()