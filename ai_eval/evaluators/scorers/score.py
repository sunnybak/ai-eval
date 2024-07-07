import asyncio

# this object contains a single score
# the score can be of any type, including a list of scores
class Score(object):
    def __init__(self, score, scorer_args=None, scorer_kwargs=None):
        self.score = score
        self.scorer_args = scorer_args
        self.scorer_kwargs = scorer_kwargs
        self.duration = None

    def __repr__(self):
        return f'{self.score}'

# scorer decorator to wrap an app scorer inside a Score object
def scorer(score_func):
    def wrapper(*scorer_args, **scorer_kwargs):
        # Check if the score function is async
        if asyncio.iscoroutinefunction(score_func):
            # If score_func is async, return a Future Score object
            async def async_wrapper():
                score = await score_func(*scorer_args, **scorer_kwargs)
                return Score(score, scorer_args, scorer_kwargs)
            return async_wrapper()
        else:
            # If score_func is sync, return a Score object
            score = score_func(*scorer_args, **scorer_kwargs)
            return Score(score, scorer_args, scorer_kwargs)
    return wrapper

