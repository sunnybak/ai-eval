class TestCase(object):

    def __init__(self, generator, scorer, target):
        self.generator = generator
        self.scorer = scorer
        self.target = target

class Dataset(object):
    def __init__(self, test_cases):
        self.test_cases = [
            TestCase(test_case[0], test_case[1], test_case[2]) for test_case in test_cases
        ]

    def __iter__(self):
        for test_case in self.test_cases:
            yield test_case

# stores the target score for evaluation
# implements the __contains__ method to check if the  is within the target
class Target(object):
    def __init__(self, target_range=[True], in_range=None):

        self.target_range = target_range
        self.custom_range = in_range is not None
        if self.custom_range:
            self._contains = lambda score: in_range(self.target_range, score)
        else:
            self._contains = self.target_range.__contains__

    def __contains__(self, score):
        if self.custom_range:
            val = self._contains(score)
            return val

        if not hasattr(score, '__iter__'):
            score = (score,)

        return self._contains(score)

    def __eq__(self, value: object) -> bool:
        return value == self.target_range
    
    def __repr__(self):
        return f'{self.target_range}'
    
