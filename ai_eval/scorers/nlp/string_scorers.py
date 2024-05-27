from nltk.util import ngrams
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import Levenshtein
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import spacy
import nltk

class StringScorer:
    def __init__(self, text, model='Alibaba-NLP/gte-large-en-v1.5'):
        self.text = text
        self.model_name = model
        self.model = None  # Model will be loaded lazily
        self.sentiment_analyzer = None  # Sentiment analyzer will be loaded lazily
        self.nlp = None  # spaCy model will be loaded lazily

    def _load_model(self):
        if self.model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def _load_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def _load_spacy_model(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # throw an exception asking the user to run - python -m spacy download en_core_web_sm
                raise Exception("spaCy model not found. Please run: python -m spacy download en_core_web_sm")

    def jaccard_similarity(self, expected, n=1):
        """
        Calculate the Jaccard similarity between the text and the expected string
        :param expected: The expected string
        :param n: The n-gram size
        :return: The Jaccard similarity between the text and the expected string
        """
        output_ngrams = set(ngrams(self.text.split(), n))
        expected_ngrams = set(ngrams(expected.split(), n))
        intersection = output_ngrams.intersection(expected_ngrams)
        union = output_ngrams.union(expected_ngrams)
        similarity = len(intersection) / len(union) if union else 0
        return similarity

    def hamming_distance(self, expected):
        """
        Calculate the Hamming distance between the text and the expected string
        :param expected: The expected string
        :return: The Hamming distance between the text and the expected string
        """
        if len(self.text) != len(expected):
            raise ValueError("Strings must be of the same length")
        distance = sum(c1 != c2 for c1, c2 in zip(self.text, expected))
        return distance

    def levenshtein_distance(self, expected):
        """
        Calculate the Levenshtein distance between the text and the expected string
        :param expected: The expected string
        :return: The Levenshtein distance between the text and the expected string
        """
        distance = Levenshtein.distance(self.text, expected)
        return distance

    def embedding_similarity(self, expected) -> float:
        """
        Calculate the embedding similarity between the text and the expected string
        :param expected: The expected string
        :return: The cosine similarity between the embeddings of the text and the expected string
        """
        self._load_model()
        sentences = [self.text, expected]
        embeddings = self.model.encode(sentences)
        similarity = cos_sim(embeddings[0], embeddings[1])
        return similarity.item()  # Convert from tensor to float

    def sentiment(self) -> float:
        """
        Calculate the sentiment score of the text using VADER
        """
        self._load_sentiment_analyzer()
        score = self.sentiment_analyzer.polarity_scores(self.text)["compound"]
        return score

    def extract_entities(self, entity_types=None):
        """
        Extract entities from the text using spaCy
        :param entity_types: List of entity types to extract. If None, all entities are extracted.
        :return: List of entities with their labels
        """
        """
        Entity type reference:
        PERSON:      People, including fictional.
        NORP:        Nationalities or religious or political groups.
        FAC:         Buildings, airports, highways, bridges, etc.
        ORG:         Companies, agencies, institutions, etc.
        GPE:         Countries, cities, states.
        LOC:         Non-GPE locations, mountain ranges, bodies of water.
        PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
        EVENT:       Named hurricanes, battles, wars, sports events, etc.
        WORK_OF_ART: Titles of books, songs, etc.
        LAW:         Named documents made into laws.
        LANGUAGE:    Any named language.
        DATE:        Absolute or relative dates or periods.
        TIME:        Times smaller than a day.
        PERCENT:     Percentage, including ”%“.
        MONEY:       Monetary values, including unit.
        QUANTITY:    Measurements, as of weight or distance.
        ORDINAL:     “first”, “second”, etc.
        CARDINAL:    Numerals that do not fall under another type.
        """
        self._load_spacy_model()
        doc = self.nlp(self.text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entitie_types:
            entities = [(text, label) for text, label in entities if label in entity_types]
        return entities

# Example usage
if __name__ == "__main__":
    # Initialize the text for the scorer
    text = "The quick brown fox jumps over the lazy dog"
    scorer = StringScorer(text)

    # Jaccard Similarity Example
    expected = "A fast brown fox leaps over a sleepy dog"
    jaccard_score = scorer.jaccard_similarity(expected, n=2)
    print(f"Jaccard Similarity Score: {jaccard_score}")

    # Hamming Distance Example
    text = "karolin"
    expected = "kathrin"
    scorer = StringScorer(text)
    hamming_score = scorer.hamming_distance(expected)
    print(f"Hamming Distance Score: {hamming_score}")

    # Levenshtein Distance Example
    text = "kitten"
    expected = "sitting"
    scorer = StringScorer(text)
    levenshtein_score = scorer.levenshtein_distance(expected)
    print(f"Levenshtein Distance Score: {levenshtein_score}")

    # Embedding Similarity Example
    text = "The quick brown fox jumps over the lazy dog"
    expected = "A fast brown fox leaps over a sleepy dog"
    scorer = StringScorer(text)
    embedding_score = scorer.embedding_similarity(expected)
    print(f"Embedding Similarity Score: {embedding_score}")

    # Sentiment Analysis Example
    text = "The movie was fantastic! I really enjoyed it."
    scorer = StringScorer(text)
    sentiment_scores = scorer.sentiment()
    print(f"Sentiment Scores: {sentiment_scores}")

    # Entity Extraction Example
    text = "Barack Obama was born in Hawaii."
    scorer = StringScorer(text)
    entities = scorer.extract_entities()
    print(f"Entities: {entities}")


__all__ = ["StringScorer"]
