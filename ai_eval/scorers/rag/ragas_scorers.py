from ai_eval.scorers.AbsBaseScorer import AbsBaseScorer
from ai_eval.scorers.score import scorer
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

class RagasScorers(AbsBaseScorer):

    @staticmethod
    @scorer
    async def ragas_faithfulness(llama_idx_query_engine_response=None):
        # print("llama_idx_query_engine_response: ", llama_idx_query_engine_response)
        assert llama_idx_query_engine_response is not None, "Response cannot be None"

        faithfulness_gpt3_5_t = FaithfulnessEvaluator()
        result = await faithfulness_gpt3_5_t.aevaluate_response(response=llama_idx_query_engine_response)
        return result.passing


    @staticmethod
    @scorer
    async def ragas_relevancy(query_str=None, llama_idx_query_engine_response=None):
        
        assert query_str is not None, "Query cannot be None"
        assert llama_idx_query_engine_response is not None, "Response cannot be None"
        
        relevancy_gpt3_5_t = RelevancyEvaluator()
        result = await relevancy_gpt3_5_t.aevaluate_response(query=query_str, response=llama_idx_query_engine_response)
        return result.passing
