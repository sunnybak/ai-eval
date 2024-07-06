from abc import ABC

class AbsBaseScorer(ABC):
    '''Base class for scorers'''
    
    def __call__(self, *args, **kwargs):
        return self._score(*args, **kwargs)

    def _score(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the score method")
    
    