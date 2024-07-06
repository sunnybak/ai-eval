import asyncio
import itertools
import pandas as pd
from typing import List, Dict, Tuple
from ai_eval.evaluators.scorers.score import Score
import time

def run_experiment(app, 
                   args: Tuple, 
                   hyperparam_dict: Dict[str, List], 
                   consistency=3
                   ):
    
    if not isinstance(args, tuple) or args is None:
        args = (None,)
    
    # post process the results after calling the app
    async def post_process(app, *args, **hyper_combo):
        start_time = time.time()
        
        # run the app
        scores = await app(*args, **hyper_combo)

        # make sure the output is a list of Score objects
        score_list = scores if isinstance(scores, list) else [scores]
        score_obj_list = [score if isinstance(score, Score) else Score(score) for score in score_list]
        
        duration = round(time.time() - start_time, 3) # milliseconds
        return score_obj_list, duration

    async def run_main():
        
        # assert that the hyperparam_dict keys are all strings and values are lists
        assert all([isinstance(k, str) for k in hyperparam_dict.keys()])
        assert all([isinstance(v, list) for v in hyperparam_dict.values()])
        
        # get the cartesian product of all hyperparam ranges
        hyperparam_kwargs = [dict(zip(hyperparam_dict, h)) for h in itertools.product(*hyperparam_dict.values())]

        # create a list of dictionaries with the args and hyperparam combinations
        results_dict = [
            {**{'arg_' + str(i): arg}, **hyper_combo}
            for hyper_combo in hyperparam_kwargs
            for i, arg in enumerate(args)
        ]
        
        # create a list of score_tasks to run the app concurrently
        score_tasks = [
            asyncio.create_task(post_process(app, *args, **hyper_combo))
            for hyper_combo in hyperparam_kwargs
            for _ in range(consistency)
        ]

        print("\nTotal number of tests:", len(score_tasks))

        # add a callback function to show the progress
        for task in score_tasks:
            task.add_done_callback(lambda fut: \
                print(f"\tTest {int(fut.get_name().split('-')[1])-1} completed"))

        # wait for all tasks to complete and unpack the results
        task_results = await asyncio.gather(*score_tasks)
        score_lists, durations = zip(*task_results)
        
        # assert that all the elements of scores have the same length
        assert all([len(score_list) == len(score_lists[0]) for score_list in score_lists])
        
        # pack the score_obj_lists into batches
        batches = []
        batch = []
        for score_obj_list, duration in zip(score_lists, durations):
            batch.append((score_obj_list, duration))
            if len(batch) == consistency:
                batches.append(batch)
                batch = []
        
        # ensure that the batches are consistent
        assert len(results_dict) == len(batches)
        
        # unpack the batches
        # batches = [ [(s1,d1), (s2,d2)], [(s3,d3), (s4,d4)] ]
        # -> [[(s1, s2), (d1, d2)], [(s3, s4), (d3, d4)]]
        # -> ([s1, s2],[s3, s4]), ([d1, d2], [d3, d4])
        # = batched_score_lists, batched_durations
        batched_score_lists, batched_durations = zip(*[list(zip(*batch)) for batch in batches])
        

        ret_results_dict = []
        num_scores = len(batched_score_lists[0][0])
        for result_dict, score_list_batch, duration_batch in \
                            zip(results_dict, batched_score_lists, batched_durations):
            
            # add duration and scores to the result_dict
            result_dict['duration'] = list(duration_batch)
            for i in range(num_scores):
                result_dict['score_' + str(i)] = []
                result_dict['scorer_args_' + str(i)] = []
                result_dict['scorer_kwargs_' + str(i)] = []

            # unpack each score list and group by batch index
            for score_list in score_list_batch:
                for i, score in enumerate(score_list):
                    result_dict['score_' + str(i)].append(score.score)
                    result_dict['scorer_args_' + str(i)].append(score.scorer_args)
                    result_dict['scorer_kwargs_' + str(i)].append(score.scorer_kwargs)

            ret_results_dict.append(result_dict)
        return pd.DataFrame(ret_results_dict)
    
    return asyncio.run(run_main())
