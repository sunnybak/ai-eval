import requests
from ai_eval.eval_result import EvalResult

# This class is used to read and write decision points to the evaluation server
# and evaluate against the decision points using scorers
class AIEval(object):
    
    EVAL_SERVER = 'http://127.0.0.1:5000'
    
    def __init__(self) -> None:
        pass

    def write_point(self, point, data):
        resp = requests.post(self.EVAL_SERVER + f'/write_point?pt={point}', json=data)
        print('Write Point: ' + resp.content.decode('utf-8'))
        return resp

    def read_point(self, point):
        resp = requests.get(self.EVAL_SERVER + f'/read_point?pt={point}')
        print('Read Point: ' + resp.content.decode('utf-8'))
        return resp.json()[point]
    
    
        
    # evaluate whether the score of the points is within the threshold
    # optionally, write the result point to the logserver
    # THOUGHT: should the result API be a separate endpoint?
    # this will allow the results to be queried separately
    # for now just dump. Later we will abstract.
    def eval(self, scorer, points, threshold, point=None):

        # fetch the data points
        datapoints = [self.read_point(p) for p in points]

        # run the scorer (can be a composed scorer)
        score = scorer(datapoints)

        # evaluate if the score is within the threshold
        # TODO: squish this into 1 line by overloading the in operator
        evaluation = score in threshold
        result = EvalResult(score, evaluation)

        # write the result to the logserver
        if point:
            # TODO: change this to json encoding of result
            self.write_point(point, str(result.score))

        return result