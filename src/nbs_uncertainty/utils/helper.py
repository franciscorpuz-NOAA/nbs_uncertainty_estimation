import configparser

def get_config(config_path: str = '../config/config.ini') -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config