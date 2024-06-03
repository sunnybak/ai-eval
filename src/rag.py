
# llama index Settings
app = RagAppSettings(
    model = '',
    embed_model = '',
    text_splitter = '',
    chunk_size = '',
    chunk_overlap = '',
    tokenizer = '',
)


evals = RagEvals(
    input_guardrails = [],
    output_guardrails = [],
    context_guardrails = [],
    context_relevance = [],
    answer_relevance = [],
    answer_faithfulness = [],
    answer_correctness = [],
)

# Query RAG
# Chat RAG

# Agent Tool Calling


tests = SynthRagUsers(
    profiles = [],
    needs = [],
    paths = [],
    styles = [],
    languages = [],
)

console = ChatConsole(app, evals, tests)
console.run() # runs until the stopping point