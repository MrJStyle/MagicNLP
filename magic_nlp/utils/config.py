import yaml

from pathlib import Path


class Config:
    def __init__(self, config_path: str):
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError("Config file not exists, please check.")

        with open(config_path) as f:
            config = yaml.safe_load(f.read())

        self.__dict__.update(config)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return None
