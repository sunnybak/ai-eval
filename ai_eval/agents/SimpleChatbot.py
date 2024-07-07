from typing import Dict, Any, Optional
from ai_eval.Agent import Agent


class SimpleChatbot(Agent):
    def process_turn(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        messages = state['messages']
        
        # Add system prompt
        messages = messages if len(messages) != 0 else [None]
        messages[0] = { 'role': 'system', 'content': self.config.system_prompt }

        # Check stop conditions
        if self.stop_condition_handler.check_turns(state):
            return None
        
        # Update state
        state['messages'] = self._generate_chat_completion(messages)
        return state