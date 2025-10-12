import json
import os

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI


def openai_login():
    load_dotenv("apis.env")
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client


def read_yaml(file_path: str):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def read_txt(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def load_config(config_type: str, name: str, experiment_path: str = None) -> dict:
    """
    Load configuration for a given experiment or fine-tuning model.

    :param config_type: Type of configuration to load ('experiments' or 'finetuning').
    :param name: Name of the experiment or fine-tuning model.
    :param experiment_path: Path to the experiment folder. If None, uses current directory.
    :return: Configuration dictionary for the specified name.
    """
    if experiment_path:
        config_file_path = os.path.join(experiment_path, "config.yaml")
    else:
        config_file_path = "config.yaml"
    
    config = read_yaml(file_path=config_file_path)

    if config_type not in config:
        print(f"Configuration type {config_type} not found in {config_file_path}.")
        exit()

    if name not in config[config_type]:
        print(
            f"{config_type.capitalize()} {name} not found in {config_file_path}. "
            f"Available {config_type}: {list(config[config_type].keys())}"
        )
        exit()

    config_args = config[config_type][name]
    return config_args


def get_experiment_paths(experiment_path: str) -> dict:
    """
    Get standardized paths for experiment folders.
    
    :param experiment_path: Base path for the experiment
    :return: Dictionary with all relevant paths
    """
    return {
        'base': experiment_path,
        'data': os.path.join(experiment_path, 'data'),
        'prompts': os.path.join(experiment_path, 'prompts'),
        'batches': os.path.join(experiment_path, 'batches'),
        'results': os.path.join(experiment_path, 'results'),
        'outputs': os.path.join(experiment_path, 'outputs'),
        'failed_experiments': os.path.join(experiment_path, 'failed_experiments'),
        'config': os.path.join(experiment_path, 'config.yaml')
    }


def ensure_experiment_directories(experiment_path: str):
    """
    Create necessary directories for an experiment if they don't exist.
    
    :param experiment_path: Base path for the experiment
    """
    paths = get_experiment_paths(experiment_path)
    for path_name, path in paths.items():
        if path_name != 'config' and path_name != 'base':  # Don't create config file or base if they don't exist
            os.makedirs(path, exist_ok=True)


def read_column_as_list(file_path: str, column_name: str) -> list[int]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return df[column_name].tolist()


def get_answers_from_results_jsonl(file_path: str) -> list[int]:
    results = []
    with open(file_path, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(
                json_object["response"]["body"]["choices"][0]["message"]["content"]
            )
    return results
