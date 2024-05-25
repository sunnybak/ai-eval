
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
    def wrapper(*args, **kwargs):
        score = score_func(*args, **kwargs)
        return Score(score, args, kwargs)
    return wrapper

