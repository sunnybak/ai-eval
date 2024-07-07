import yaml
from typing import Dict, Any, List, Callable, Optional
import importlib
from dataclasses import dataclass

@dataclass
class ModelSettings:
    model: str
    api_key: str
    hyperparams: Dict[str, Any]

@dataclass
class AgentConfig:
    architecture: Any
    model_settings: ModelSettings
    input: str
    state: str
    system_prompt: str
    stop_conditions: Dict[str, Any]
    role: str

@dataclass
class AppConfig:
    agent: AgentConfig

@dataclass
class EvaluatorConfig:
    input: Optional[str] # default to 'state.messages'
    params: Optional[Dict[str, Any]]
    model_settings: Optional[ModelSettings]

@dataclass
class SimulationConfig:
    agent: AgentConfig
    personas: Dict[str, Any]
    scenarios: Dict[str, Any]

@dataclass
class AIEvalConfig:
    app: AppConfig
    evaluators: Dict[str, EvaluatorConfig]
    simulations: SimulationConfig

class ModuleLoader:
    @staticmethod
    def load_class(package: str, module_name: str):
        try:
            module = importlib.import_module('ai_eval.' + package + '.' + module_name, package='ai_eval')
            if hasattr(module, module_name):
                return getattr(module, module_name)
            else:
                raise AttributeError(f"Module {module} has no class named {module_name}")
        except ImportError:
            raise ImportError(f"Could not import module {module_name} from package {package}")

class YAMLInterpreter:
    def __init__(self, yaml_file: str):
        self.yaml_file = yaml_file
        self.raw_config: Dict[str, Any] = {}
        self.ai_eval_config: Optional[AIEvalConfig] = None
        
        ### 
        self.app_agent: Dict[str, Any] = {}
        self.evaluators: Dict[str, Any] = {}
        self.simulation_agent: Dict[str, Any] = {}
        
    @staticmethod
    def initialize_agent(agent_config: AgentConfig, agent_architecture) -> Any:
        agent_class = ModuleLoader.load_class('agents', agent_architecture)
        return agent_class(agent_config)

    def load_yaml(self):
        with open(self.yaml_file, 'r') as file:
            self.raw_config = yaml.safe_load(file)

    def parse_model_settings(self, config: Dict[str, Any]) -> ModelSettings:
        return ModelSettings(
            model=config.get('model', ''),
            api_key=config.get('api_key', ''),
            hyperparams=config.get('hyperparams', {})
        )

    def parse_agent_config(self, config: Dict[str, Any]) -> AgentConfig:
        return AgentConfig(
            architecture=config.get('architecture', ''),
            model_settings=self.parse_model_settings(config.get('model_settings', {})),
            state=config.get('state', ''),
            input=config.get('input', ''),
            system_prompt=config.get('system_prompt', ''),
            stop_conditions=config.get('stop_conditions', {}),
            role=config.get('role', 'assistant')
        )

    def load_app_config(self) -> AppConfig:
        app_config = self.raw_config.get('app', {})
        agent_config = self.parse_agent_config(app_config.get('agent', {}))
        agent_config.role = 'assistant'

        # Load simulation modules
        self.app_agent = YAMLInterpreter.initialize_agent(agent_config, agent_config.architecture)

        return AppConfig(agent=agent_config)

    def load_evaluators_config(self) -> Dict[str, EvaluatorConfig]:
        evaluators_config = self.raw_config.get('evaluators', {})
        parsed_evaluators = {}
        
        for name, config in evaluators_config.items():
            config = config or {}
            parsed_evaluators[name] = EvaluatorConfig(
                input=config.get('input', ''),
                model_settings=config.get('model_settings', {}),
                params=config.get('params', {}),
            )
            # Load evaluator modules
            self.evaluators[name] = ModuleLoader.load_class('evaluators', name)
        
        return parsed_evaluators

    def load_simulations_config(self) -> SimulationConfig:
        simulations_config = self.raw_config.get('simulations', {})
        agent_config = self.parse_agent_config(simulations_config.get('agent', {}))
        agent_config.role = 'user'
        
        # Load simulation modules
        self.simulation_agent = YAMLInterpreter.initialize_agent(agent_config, agent_config.architecture)
        
        return SimulationConfig(
            agent=agent_config,
            personas=simulations_config.get('personas', {}),
            scenarios=simulations_config.get('scenarios', {})
        )

    def initialize(self):
        self.load_yaml()
        app_config = self.load_app_config()
        evaluators_config = self.load_evaluators_config()
        simulations_config = self.load_simulations_config()
        
        self.ai_eval_config = AIEvalConfig(
            app=app_config,
            evaluators=evaluators_config,
            simulations=simulations_config
        )

    def get_raw_config(self) -> Dict[str, Any]:
        return self.raw_config

    def get_ai_eval_config(self) -> Optional[AIEvalConfig]:
        return self.ai_eval_config

    def get_app_agent(self):
        return self.app_agent

    def get_evaluators(self) -> Dict[str, Any]:
        return self.evaluators

    def get_simulation_app(self) -> Dict[str, Any]:
        return self.simulation_agent