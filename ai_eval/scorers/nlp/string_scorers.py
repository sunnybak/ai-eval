from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from ai_eval.scorers.AbsBaseScorer import AbsBaseScorer

class EmbeddingSimilarityScorer(AbsBaseScorer):
    def __init__(self, model='Alibaba-NLP/gte-large-en-v1.5'):
        self.model = SentenceTransformer(model, trust_remote_code=True)
    
    def compute_score(self, output, expected):
        sentences = [output, expected]
        embeddings = self.model.encode(sentences)
        similarity = cos_sim(embeddings[0], embeddings[1])
        return similarity.item()  # Convert from tensor to float
    
    def __call__(self, output, expected):
        score = self.compute_score(output, expected)
        return score 

# Example usage
if __name__ == "__main__":
    scorer = EmbeddingSimilarityScorer()
    output = "The quick brown fox jumps over the lazy dog"
    expected = "A fast brown fox leaps over a sleepy dog"
    score = scorer(output, expected)
    print(f"Embedding Similarity Score: {score}")

