from pathlib import Path
import yaml

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def read_yaml(path):
    with open(path, "r") as input_file:
        dict_yaml = yaml.safe_load(input_file)

    return dict_yaml
