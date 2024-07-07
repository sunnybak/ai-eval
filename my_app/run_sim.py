
from ai_eval import YAMLInterpreter
from ai_eval.agents import SimpleChatbot

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_chat(messages):
    for m in messages:
        if m["role"] == 'user':
            print(bcolors.OKBLUE + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)
        elif m["role"] == 'assistant':
            print(bcolors.OKGREEN + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)
        elif m["role"] == 'system':
            print(bcolors.HEADER + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)

def main():
    simulator = YAMLInterpreter('my_app/config.yaml')
    simulator.initialize()
    
    state = {'messages': []}

    for i in range(3):
        simulator.simulation_agent.process_turn(state)
        simulator.app_agent.process_turn(state)

    print_chat(state['messages'])
    
    print(simulator.evaluators['message_counter'](state))


if __name__ == '__main__':
    main()