# this file contains some helpful generator templates

def simple_trajectory_generator(messages, simple_trajectory):
    """
    A simple trajectory generator that generates a fixed trajectory
    - we ignore the AI responses and only consider the user messages

    :param messages: the messages in the conversation so far
    :param simple_trajectory: the simple trajectory to be executed
    :return: the next message in the trajectory 
    """
    user_idx = len(list(filter(lambda x: x['role'] == 'user', messages)))

    if user_idx < len(simple_trajectory):
        return {
            "role": "user",
            "content": simple_trajectory[user_idx]
        }

    return None

__all__ = ['simple_trajectory_generator']
