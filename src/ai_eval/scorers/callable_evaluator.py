from abc import ABC, abstractmethod

# a wrapper around a callable evaluator function
class CallableEvaluator(ABC):
    
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("evaluate method must be implemented")
    
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)