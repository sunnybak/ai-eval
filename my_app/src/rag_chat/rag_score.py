# Getting lamma to work with us
from llama_index.core  import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

import random 

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken



Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)


from ai_eval.scorers.rag import RagasScorers
from ai_eval import run_experiment

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

eval_questions_all = []
num_questions_per_chunk = 1


def load_eval_documents(data_dir, n=10):
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    eval_documents = [documents[random.randint(0, len(documents)-1)] for _ in range(n)]

    return eval_documents

def generate_questions(eval_documents):
    
    # naively generate questions
    data_generator = RagDatasetGenerator.from_documents(eval_documents)
    
    # generate questions based on query
    # q_gen_query = f"You are a scientific researcher. \
    #         Your task is to setup {num_questions_per_chunk} questions. \
    #         The questions must be related to following \
    #         1. my interest 1 2.My interest 2 3. My interest 3 \
    #         Restrict the questions to the context information provided."
    # data_generator = RagDatasetGenerator.from_documents(eval_documents,  
    #               question_gen_query=q_gen_query)

    eval_questions = data_generator.generate_questions_from_nodes()
    
    eval_questions_all = []
    eval_questions_all.append(eval_questions.to_pandas()['query'].to_list())
    questions = eval_questions.to_pandas()['query'].to_list() 
    return questions


async def run_app(questions, chunk_size, model):
    Settings.llm = OpenAI(model=model, temperature=0, chunk_size=chunk_size)
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = round(chunk_size/10,0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])
    
    # embed the documents
    vector_index = VectorStoreIndex.from_documents(eval_documents)

    # Create a query engine
    query_engine = vector_index.as_query_engine()

    faithfulness_scores = []
    relevancy_scores = []
    for question in questions:
        response_vector = query_engine.query(question)
    
        faithfulness_scores.append(await RagasScorers.ragas_faithfulness(llama_idx_query_engine_response=response_vector))
        relevancy_scores.append(await RagasScorers.ragas_relevancy(query_str=question, llama_idx_query_engine_response=response_vector))
    
    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )
    token_counter.reset_counts()
    return [faithfulness_scores, relevancy_scores]


# run_app: function(*args, **kwargs) -> Report({scores: List[Score], cost: CostDict})


if __name__ == "__main__":
    
    from ai_eval.scorers.nlp.token_scores import TokenScorer

    eval_documents = load_eval_documents("my_app/src/rag_chat/data")
    # questions = generate_questions(eval_documents)
    
    # embedding_tokens = sum([TokenScorer.token_count(doc) for doc in eval_documents])
    # embedding_cost = 0.02*10**-6 * embedding_tokens
    
    questions = [
        'What is yo mamma',
        'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 
        'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 
        'What visual vignettes are described in the Bone and calcium disorders section of the journal, and what conditions do they illustrate?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports publication for the year 2024?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, as described in one of the case reports?', 'In what ways did a patient with Type 1 Diabetes Mellitus on insulin experience hypoglycemia unawareness and recurrent severe hypoglycemia, as detailed in a case report in the document?', 'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 'In the Bone and calcium disorders section, what are some of the interesting visual vignettes described in the current issue of AACE Clinical Case Reports?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports, as mentioned in the provided text?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, according to the text?', 'In what ways did a patient with Type 1 Diabetes Mellitus on insulin experience hypoglycemia unawareness and recurrent severe hypoglycemia, as described in the AACE Clinical Case Reports?', 'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 'What visual vignettes are described in the Bone and calcium disorders section of the journal, and what conditions do they illustrate?', 'What is the title of the journal where the clinical case reports are published in the given context information?', 'Describe a case study mentioned in the document that involves a transgender woman and a unique medical condition.', 'How is the publication licensed according to the context information provided?', 'What are some of the interesting cases highlighted in the current issue of AACE Clinical Case Reports (ACCR) related to adrenal and neuroendocrine disorders?', 'In the area of transgender care, what specific case was highlighted in this issue of ACCR and where can listeners find a podcast summarizing the significance of this case?', 'How do authors in the field of Diabetes and Metabolism suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes, as discussed in this issue of ACCR?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports publication for the year 2024?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, as described in one of the case reports?', 'In what ways did a patient with Type 1 Diabetes Mellitus on insulin experience hypoglycemia unawareness and recurrent severe hypoglycemia, as detailed in a case report in the document?', 'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 'What visual vignettes are described in the Bone and calcium disorders section of the journal, and what do they focus on?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports, as mentioned in the provided text?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, according to the text?', 'In what ways did a patient with Type 1 Diabetes Mellitus on insulin experience hypoglycemia unawareness and recurrent severe hypoglycemia, as described in the AACE Clinical Case Reports?', 'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 'What visual vignettes are described in the Bone and calcium disorders section of the journal, and what conditions do they illustrate?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports publication for the year 2024?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, as described in one of the case reports?', 'In the case of a transgender woman experiencing hematospermia, what evidence was found for endometrial tissue in the prostate, as detailed in the AACE Clinical Case Reports?', 'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 'What visual vignettes are described in the Bone and calcium disorders section of the journal, and what do they focus on?', 'What is the title of the journal where the clinical case reports are published in the given context information?', 'Describe a case study mentioned in the document that involves a transgender woman and hematospermia.', 'How is the publication licensed according to the context information provided?', 'What are some of the interesting cases highlighted in the current issue of AACE Clinical Case Reports (ACCR) related to adrenal and neuroendocrine disorders?', 'In the area of transgender care, what specific case is discussed in this issue of ACCR and where can listeners find a podcast summarizing its significance?', 'How do authors in the field of Diabetes and Metabolism suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes, as mentioned in the editorial?', 'What is the title of the journal where the clinical case reports are published in the given context information?', 'Describe one case study mentioned in the document that involves a transgender woman.', 'How is the open access article in the given context information licensed for distribution?', 'What are some of the interesting cases highlighted in the current issue of AACE Clinical Case Reports (ACCR)?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section?', 'What are some of the adverse events associated with immune checkpoint inhibitors in cancer patients as reported in the current issue of ACCR?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports publication for the year 2024?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, as described in one of the case reports?', 'In the case of a transgender woman experiencing hematospermia, what evidence was found for endometrial tissue in the prostate, as mentioned in the AACE Clinical Case Reports?', 'What are some of the rare adrenal and neuroendocrine cases highlighted in the current issue of AACE Clinical Case Reports?', 'How do authors suggest minimizing Hypoglycemia unawareness in patients with type 1 diabetes in the Diabetes and Metabolism section of the journal?', 'What visual vignettes are described in the Bone and calcium disorders section of the journal, and what do they focus on?', 'What are some of the rare medical cases discussed in the AACE Clinical Case Reports, as mentioned in the provided text?', 'How did immune checkpoint inhibitor therapy trigger severe hyponatremia in a patient with Mulvihill-Smith Syndrome, according to the text?', 'In what ways did a patient with Type 1 Diabetes Mellitus on insulin experience hypoglycemia unawareness and recurrent severe hypoglycemia, as described in the text?'
    ]
    
    # COST

    # doc_size_tokens: 1000
    # num_docs: 10
    
    # O(n): n = number of questions per run
    
    # MVP
    # cost calculator
    # chat RAG on finance dataset
    

    experiment_summary = run_experiment(app=run_app,
                           args=(questions[:1],),
                           hyperparam_dict={
                               'chunk_size': [256], # embedding_cost + inference_cost(n_questions * (chunk_size + query + response))  0.00768
                               'model': ['gpt-3.5-turbo'], 
                            },
                           consistency=3,
                           )

    print(experiment_summary.columns)
    print(experiment_summary[['chunk_size','score_0', 'score_1']])
    
    print(experiment_summary[['scorer_kwargs_0', 'scorer_kwargs_1']])
    
    # virtualenv venv
    # source venv/bin/activate
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size
# def evaluate_response_time_and_accuracy(eval_documents, chunk_size, eval_questions):
#     total_response_time = 0
#     total_faithfulness = 0
#     total_relevancy = 0

#     response_vectors = []

#     # Update settings during each run 
#     Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, chunk_size = chunk_size )
#     Settings.chunk_size = chunk_size
#     Settings.chunk_overlap = round(chunk_size/10,0)
#     Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    
#     # embed the documents
#     vector_index = VectorStoreIndex.from_documents(eval_documents)

#     # Create a query engine
#     query_engine = vector_index.as_query_engine()
#     num_questions = len(eval_questions)

#     for question in eval_questions:
#         start_time = time.time() 
#         # Generate a response vector
#         response_vector = query_engine.query(question)
#         print('Question:', question)
#         print('Response:', str(response_vector))

#         elapsed_time = time.time() - start_time
        
#         # Evaluate the quality of response 
#         faithfulness_result = RagasScorers.ragas_faithfulness(llama_idx_query_engine_response=response_vector)
#         relevancy_result = RagasScorers.ragas_relevancy(query_str=question, llama_idx_query_engine_response=response_vector)
        
#         # Document the quality of resposne
#         response_vectors.append({"chunk_size" : chunk_size,
#                                  "question" : question,
#                                  "response_vector" : response_vector,
#                                  "faithfulness_result" : faithfulness_result,
#                                  "relevancy_result" : relevancy_result})

#         total_response_time += elapsed_time
#         total_faithfulness += faithfulness_result
#         total_relevancy += relevancy_result
    
#     # Get average score over all questions
#     average_response_time = total_response_time / num_questions
#     average_faithfulness = total_faithfulness / num_questions
#     average_relevancy = total_relevancy / num_questions

#     return average_response_time, average_faithfulness, average_relevancy, response_vectors
# # response_vectors_all = []
# # for chunk_size in [128]:
# #     avg_time, avg_faithfulness, avg_relevancy, response_vectors = evaluate_response_time_and_accuracy(eval_documents, chunk_size, questions)
# #     [response_vectors_all.append(i) for i in response_vectors]
# #     print(f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
# #     time.sleep(5)
    
    