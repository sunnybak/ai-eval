from ai_eval.evaluators.scorers import AbsBaseScorer
import tiktoken

class TokenScorer(AbsBaseScorer):
    
    DEFAULT_ENCODER_FOR_MODEL = "gpt-4"
    
    def __init__(self, encoder_model=None):
        encoder_model = encoder_model or TokenScorer.DEFAULT_ENCODER_FOR_MODEL
        self._encoding = tiktoken.encoding_for_model(encoder_model)
        super().__init__()
        
    def token_encode_text(self, text: str) -> list:
        assert isinstance(text, str)
        return self._encoding.encode(text)
        
    def token_count(self, text: str) -> int:
        token_ints = self.token_encode_text(text)
        return len(token_ints)

    def token_decode_text(self, token_ints: list, encoding_format=None) -> str:
        token_bytes = [self._encoding.decode_single_token_bytes(token) for token in token_ints]
        # return bytes if no encoding provided. Otherwise, decode to utf-8 string
        if encoding_format is None:
            return b''.join(token_bytes)
        return b''.join(token_bytes).decode(encoding_format)

if __name__ == '__main__':
    
    base_scorer = TokenScorer()
    
    text = "artificial intelligence evaluation"
    token_ints = base_scorer.token_encode_text(text)
    print(token_ints)
    print(base_scorer.token_decode_text(token_ints))
    print(base_scorer.token_decode_text(token_ints, 'utf-8'))
    print(base_scorer.token_count(text))
