from nltk.util import ngrams
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import Levenshtein
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import pipeline
import spacy
import nltk
import textstat
from langdetect import detect

class StringScorer:
    def __init__(
             self, 
             text, 
             embedding_model='Alibaba-NLP/gte-large-en-v1.5',
             classification_model='valhalla/distilbart-mnli-12-9'
        ):
        self.text = text
        self.embedding_model_name = embedding_model
        self.classification_model_name = classification_model
        self.embedding_model = None  # Model will be loaded lazily
        self.classification_model = None # Model will be loaded lazily
        self.sentiment_analyzer = None  # Sentiment analyzer will be loaded lazily
        self.nlp = None  # spaCy model will be loaded lazily

    def _load_embedding_model(self):
        if self.embedding_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)

    def _load_classification_model(self):
        if self.classification_model is None:
            self.classification_model = pipeline("zero-shot-classification", model=self.classification_model_name)

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

    def language(self):
        """
        Detect the language of the text
        :return: The language of the text
        """
        lang = detect(self.text)
        return lang

    def readability(self, precision=0, lang="en") -> float:
        """
        Calculate the grade level of the text
        :param precision: The number of decimal places to round the grade level to
        :param lang: The language of the text - can be "en", "es", "ar", "de", "it" 
        :return: Which grade level can read the text
        """
        grade_level = 0
        if lang == "en":
            grade_level = textstat.text_standard(self.text, float_output=True)
        elif lang == "es":
            grade_level = textstat.szigriszt_pazos(self.text)
        elif lang == "ar":
            grade_level = textstat.osman(self.text)
        elif lang == "de":
            grade_level = textstat.wiener_sachtextformel(self.text)
        elif lang == "it":
            grade_level = textstat.gulpease_index(self.text)
        else:
            # throw not implemented
            raise Exception("Language not supported")
        grade_level = round(grade_level, precision)
        return grade_level

    def reading_time(self, wpm=200) -> float:
        """
        Calculate the reading time of the text
        :param wpm: The words per minute to use for the calculation
        :return: The reading time of the text in seconds
        """
        ms_per_char = 60000 / wpm
        reading_time = textstat.reading_time(self.text, ms_per_char=ms_per_char)
        return reading_time

    def topics(self, topics=None, threshold=0.5):
        if topics is None:
            raise ValueError("Topics must be provided")
        self._load_classification_model()
        if self.classification_model is None:
            raise Exception("Model not found")
        result = self.classification_model(self.text, topics)
        result = {topic: score for topic, score in zip(result["labels"], result["scores"])}
        result = {topic: score for topic, score in result.items() if score > threshold}
        # turn into a list
        result = list(result.keys())
        return result

    def sentiment(self) -> float:
        """
        Calculate the sentiment score of the text using VADER
        """
        self._load_sentiment_analyzer()
        score = self.sentiment_analyzer.polarity_scores(self.text)["compound"]
        return score

    def entities(self, entity_types=None):
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
        if self.nlp is None:
            raise Exception("Spacy model not found, have you installed the model correctly?")
        doc = self.nlp(self.text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entity_types:
            entities = [(text, label) for text, label in entities if label in entity_types]
        return entities

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
        self._load_embedding_model()
        sentences = [self.text, expected]
        if self.embedding_model is None:
            raise Exception("Embedding model not found")
        embeddings = self.embedding_model.encode(sentences) 
        similarity = cos_sim(embeddings[0], embeddings[1])
        return similarity.item()  # Convert from tensor to float

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
    entities = scorer.entities()
    print(f"Entities: {entities}")

    # Readability Example
    text = "The quick brown fox jumps over the lazy dog"
    scorer = StringScorer(text)
    grade_level = scorer.readability()
    print(f"Grade Level: {grade_level}")

    # Reading Time Example
    text = "The quick brown fox jumps over the lazy dog"
    scorer = StringScorer(text)
    reading_time = scorer.reading_time()
    print(f"Reading Time: {reading_time}")

    # Topics Example
    text = "The quick brown fox jumps over the lazy dog"
    topics = ["animals", "nature"]
    scorer = StringScorer(text)
    topic_scores = scorer.topics(topics)
    print(f"Topic Scores: {topic_scores}")

__all__ = ["StringScorer"]
