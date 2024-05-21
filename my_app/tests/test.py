# contains the result of an evaluation
# it stores the full breakdown of the evaluation
# it can perform statistical analysis on the results
class Result(object):
    def __init__(self, result):
        self.result = result
    
def score(x): return x

def in_range(range, data):
    return all(x in range for x in data)

data = [3,5,7]

target = Target(range=[1, 2, 4, 3,5,7], in_range=in_range)
result = score(data) in target

print(result)
print(3 in target)
print([1, 2, 4, 3, 5, 7] == target)
# eval = ai.eval(scorer=score, target=Target([1, 2, 3, 4, 5])

