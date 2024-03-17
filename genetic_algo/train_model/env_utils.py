import re
import math
from environment_abstract import Environment


def get_environment(env_name: str) -> Environment:
    env_name = env_name.lower()
    env: Environment

    from cube3 import Cube3
    env = Cube3()
    
    return env
