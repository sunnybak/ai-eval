from enum import Enum

class ChatRoles(Enum):
    '''OpenAI chat roles'''
    system = 'system'
    user = 'user'
    assistant = 'assistant'
    
class ChatKeys(Enum):
    '''Keys for OpenAI chat messages'''
    role = 'role'
    content = 'content'
