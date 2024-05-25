from typing import Optional, Any, Callable

# a callable evaluator class
# has 2 components: a pass_range, and an in_range function
class BaseEvaluator(object):
    
    DEFAULT_PASS_RANGE: Optional[Any] = True
    DEFAULT_IN_RANGE: Callable[[Any, Any], bool] \
        = lambda pass_range, score: score == BaseEvaluator.DEFAULT_PASS_RANGE

    def __init__(self, 
                 pass_range: Optional[Any] = None,
                 in_range: Callable[[Any, Any], bool] = None):

        self._pass_range = pass_range or BaseEvaluator.DEFAULT_PASS_RANGE

        if in_range is not None:
            self._in_range = in_range
        elif pass_range is not None:
            self._in_range = lambda pass_range, score: score == pass_range
        else:
            self._in_range = BaseEvaluator.DEFAULT_IN_RANGE
    
    # calling the evaluator runs the in_range function on the pass_range
    def __call__(self, *in_range_args, **in_range_kwargs):
        return self._in_range(self._pass_range, *in_range_args, **in_range_kwargs)
    
    # run the eval over a custom range with the same in_range function
    def in_custom_range(self, custom_range, *in_range_args, **in_range_kwargs):
        assert custom_range is not None, 'custom_range must be provided'
        return BaseEvaluator(custom_range, self._in_range, *in_range_args, **in_range_kwargs)

    # run the eval over the same range with a custom in_range function
    def with_custom_in_range(self, custom_in_range, *in_range_args, **in_range_kwargs):
        assert custom_in_range is not None, 'custom_in_range must be provided'
        return BaseEvaluator(self._pass_range, custom_in_range, *in_range_args, **in_range_kwargs)

    # check if a score is in the pass_range
    # example: 0.8 in eval -> True
    def __contains__(self, score):
        return self._in_range(self._pass_range, score)

    def __repr__(self) -> str:
        return f'{self._pass_range}'
