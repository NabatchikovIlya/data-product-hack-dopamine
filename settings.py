from dotenv import load_dotenv
from environs import Env

load_dotenv()
env = Env()


class Settings:
    MODEL_NAME = env.str('MODEL_NAME', 'dph-dopamine')
