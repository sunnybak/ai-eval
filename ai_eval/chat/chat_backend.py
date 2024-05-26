
    

class ChatBackend():
    
    def __init__(self, model='gpt-3.5-turbo', sys_prompt=None):
        self._model = model
        self._sys_prompt = sys_prompt
        
    def get_response(self, text):
        return self.chat.get_response(text)
