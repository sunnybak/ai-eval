from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from litellm import completion

@dataclass
class ModelSettings:
    model: str
    api_key: str
    hyperparams: Dict[str, Any]

@dataclass
class AgentConfig:
    architecture: str
    model_settings: ModelSettings
    input: str
    system_prompt: str
    stop_conditions: Dict[str, Any]

class StopConditionHandler:
    def __init__(self, stop_conditions: Dict[str, Any]):
        self.stop_conditions = stop_conditions
        self.current_tokens = 0
        self.current_inputs = 0
        
    def check_turns(self, state: Dict[str, Any]):
        return len(state.get('messages')) >= self.stop_conditions.get('max_turns', float('inf'))

class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.stop_condition_handler = StopConditionHandler(config.stop_conditions)

    def _generate_chat_completion(self, messages: Any) -> str:
        
        # swap roles for user
        if self.config.role == 'user':
            messages = self.swap_roles(messages)

        response = completion(
            model=self.config.model_settings.model,
            api_key=self.config.model_settings.api_key,
            messages=messages,
            **self.config.model_settings.hyperparams
        )
        
        # undo swap roles for user
        if self.config.role == 'user':
            messages = self.swap_roles(messages)

        # get response and add the right role
        response = response.choices[0].message
        response['role'] = self.config.role
        messages.append(response)

        return messages

    def swap_roles(self, messages: Any) -> Any:
        for message in messages:
            if message['role'] == 'user':
                message['role'] = 'assistant'
            elif message['role'] == 'assistant':
                message['role'] = 'user'
        return messages

    @abstractmethod
    def process_turn(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a turn based on the current state.
        
        :param state: The current state of the conversation or task.
        :return: A new state if processing should continue, or None if it should stop.
        """
        pass

    def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        current_state = initial_state
        while True:
            new_state = self.process_turn(current_state)
            if new_state is None:
                break
            current_state = new_state
        return current_state

