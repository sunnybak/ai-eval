from ai_eval.scorers import AbsBaseScorer
from ai_eval.scorers.nlp.token_scores import TokenScorer
from ai_eval.scorers.chat.chat_types import ChatRoles, ChatKeys
from typing import Dict


class ChatScorer(AbsBaseScorer):
    '''Scorer for chat tasks'''
    
    @staticmethod
    def validate_openai_format(messages):
        '''Validates that the messages are in the OpenAI format'''

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary")
            
            # assert that all message keys = ChatKeys
            if not all([key in ChatKeys.__members__ for key in message]):
                raise ValueError(f"Each message must have the following keys: {ChatKeys.__members__.values()}")

            # assert that the role is one of the ChatRoles
            assert message['role'] in ChatRoles.__members__, f"Role must be one of {ChatRoles.__members__.values()}"

    def __init__(self, messages):
        # store the messages
        ChatScorer.validate_openai_format(messages)
        self.messages = messages
        
        # token scorer
        self._token_scorer = TokenScorer()

        super().__init__()

    @staticmethod
    def total_chat_length(self) -> int:
        return len(self.messages)
    
    @staticmethod
    def chat_role_counts(self) -> Dict[ChatRoles, int]:
        '''Returns the number of system, user, assistant messages'''
        
        msg_counts = {'user': 0, 'assistant': 0, 'system': 0}
        for message in self.messages:
            msg_counts[message['role']] += 1
        return msg_counts
        
    @staticmethod
    def chat_role_tokens(self, encoder_model=None) -> Dict[ChatRoles, int]:
        '''Returns the number of tokens per role in the chat'''

        self._token_scorer = self._token_scorer or TokenScorer(encoder_model)
        token_counts = {'user': 0, 'assistant': 0, 'system': 0}
        for message in self.messages:
            token_counts[message['role']] += self._token_scorer.token_count(message['content'])

        return token_counts
    
    def chat_total_tokens(self, encoder_model=None) -> Dict[ChatRoles, int]:
        '''Returns the number of tokens per role in the chat'''

        self._token_scorer = self._token_scorer or TokenScorer(encoder_model)
        token_counts = {'input': 0, 'output': 0}
        running = 0
        for message in self.messages:
            new_input = self._token_scorer.token_count(message['content'])
            if message['role'] == 'user' or message['role'] == 'system':
                token_counts['input'] += new_input + running
            elif message['role'] == 'assistant':
                token_counts['output'] += new_input
            running += new_input
        return token_counts    
    
    @staticmethod
    def next_turn(self) -> ChatRoles:
        '''Returns the next turn in the chat'''

        if self.messages[-1]['role'] == ChatRoles.user:
            return ChatRoles.assistant
        return ChatRoles.user
