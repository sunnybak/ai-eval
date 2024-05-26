

class ChatClient():
    
    def __init__(self, model='gpt-3.5-turbo'):
        self._model = model
        self._chat = None
    
    