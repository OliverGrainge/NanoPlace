import yaml


def read_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config
