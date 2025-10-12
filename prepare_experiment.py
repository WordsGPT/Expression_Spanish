"""
This script prepares tasks for psycholinguistic experiments using the OpenAI API.
It reads configuration from config.yaml and generates batched JSONL files for execution.

Usage:
    python prepare_experiment.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>
    
Arguments:
    - <experiment_name>: Process specific experiment
    - <experiment_path>: Path to experiment folder (where config.yaml, data/, prompts/ are located)
    - all: Process all experiments  
    - failed: Retry failed experiments
    - status: Show failed experiments status
"""

import os
import sys
import jsonlines
from datetime import datetime
import pandas as pd
import yaml
import re
from utils import get_experiment_paths, ensure_experiment_directories, load_config, read_yaml


def load_word_list(file_path: str, column_name: str) -> list:
    """Load word list from CSV or Excel file."""
    print(f"Loading word list from {file_path}, column '{column_name}'...")
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for file: {file_path}")
        
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file extension: {file_path}")
        
        if column_name not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Column '{column_name}' not found. Available: {available_columns}")
        
        word_list = df[column_name].dropna().astype(str).tolist()
        print(f"Successfully loaded {len(word_list)} expressions.")
        return word_list
    
    except Exception as e:
        print(f"Error loading word list: {e}")
        raise


def load_prompt_from_file(prompt_file: str, experiment_path: str) -> str:
    """Load prompt from text file."""
    paths = get_experiment_paths(experiment_path)
    prompt_path = os.path.join(paths['prompts'], prompt_file)
    
    try:
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_content = file.read().strip()
        
        if not prompt_content:
            raise ValueError(f"Prompt file is empty: {prompt_path}")
        
        print(f"Loaded prompt from {prompt_file}")
        return prompt_content
    
    except Exception as e:
        print(f"Error loading prompt file: {e}")
        raise


def create_openai_tasks(word_list: list, config: dict, experiment_name: str, experiment_path: str) -> list:
    """Create OpenAI API tasks from word list and configuration."""
    prompt_file = config.get('prompt', '')
    if not prompt_file:
        raise ValueError(f"No prompt file specified in config for experiment {experiment_name}")
    
    prompt_template = load_prompt_from_file(prompt_file, experiment_path)
    model_name = config.get('model_name', 'gpt-4o-mini')
    
    # Find the placeholder in the prompt (text between [ ])
    placeholder_match = re.search(r'\[([^\]]+)\]', prompt_template)
    
    if placeholder_match:
        prompt_key = f"[{placeholder_match.group(1)}]"
    else:
        prompt_key = config.get('prompt_key', '[insertar expresión aquí]')
    
    tasks = []
    for counter, expression in enumerate(word_list, start=1):
        prompt_content = prompt_template.replace(prompt_key, str(expression).strip())
        
        task = {
            "custom_id": f"{experiment_name}_task_{counter}",
            "method": "POST", 
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 10,
                "response_format": {"type": "text"},
                "messages": [
                    {"role": "user", "content": prompt_content}
                ]
            }
        }
        tasks.append(task)
    
    print(f"Created {len(tasks)} tasks for experiment {experiment_name}")
    return tasks


def create_batch_files(tasks: list, experiment_name: str, experiment_path: str, chunk_size: int = 50000) -> list:
    """Split tasks into batches and save as JSONL files."""
    paths = get_experiment_paths(experiment_path)
    
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    batch_files = []
    for index, chunk in enumerate(task_chunks):
        batch_name = f"{experiment_name}_batch_{index}_{date_string}.jsonl"
        batch_path = os.path.join(paths['batches'], batch_name)
        
        with jsonlines.open(batch_path, "w") as file:
            file.write_all(chunk)
        
        batch_files.append(batch_name)
        print(f"Created batch file: {batch_name} with {len(chunk)} tasks")
    
    return batch_files


def load_all_configs(experiment_path: str) -> dict:
    """Load all experiment configurations from config.yaml."""
    paths = get_experiment_paths(experiment_path)
    
    try:
        if not os.path.exists(paths['config']):
            raise FileNotFoundError(f"Config file not found at {paths['config']}")
        
        config_data = read_yaml(paths['config'])
        experiments = config_data.get('experiments', {})
        
        if not experiments:
            raise ValueError("No experiments found in config.yaml")
        
        return experiments
        
    except FileNotFoundError:
        print(f"Error: config.yaml file not found at {paths['config']}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        sys.exit(1)


def load_failed_experiments(experiment_path: str) -> list:
    """Load list of failed experiments from file."""
    paths = get_experiment_paths(experiment_path)
    failed_file = os.path.join(paths['failed_experiments'], "failed_prepare_exp.txt")
    try:
        if os.path.exists(failed_file):
            with open(failed_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not read failed experiments file: {e}")
    return []


def save_failed_experiments(failed_list: list, experiment_path: str):
    """Save list of failed experiments to file."""
    paths = get_experiment_paths(experiment_path)
    failed_file = os.path.join(paths['failed_experiments'], "failed_prepare_exp.txt")
    
    try:
        if not failed_list:
            if os.path.exists(failed_file):
                os.remove(failed_file)
            if os.path.exists(paths['failed_experiments']) and not os.listdir(paths['failed_experiments']):
                os.rmdir(paths['failed_experiments'])
            return
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed_list) + '\n')
    except Exception as e:
        print(f"Warning: Could not save failed experiments file: {e}")


def update_failed_experiments(experiment_name: str, success: bool, experiment_path: str):
    """Add or remove experiment from failed list based on success."""
    failed_experiments = load_failed_experiments(experiment_path)
    
    if success and experiment_name in failed_experiments:
        failed_experiments.remove(experiment_name)
    elif not success and experiment_name not in failed_experiments:
        failed_experiments.append(experiment_name)
    
    save_failed_experiments(failed_experiments, experiment_path)


def process_experiment(experiment_name: str, config: dict, experiment_path: str) -> bool:
    """Process a single experiment and return success status."""
    print(f"\n{'='*60}")
    print(f"Processing experiment: {experiment_name}")
    print(f"{'='*60}")
    
    try:
        paths = get_experiment_paths(experiment_path)
        
        required_fields = ['dataset_name', 'prompt', 'model_name']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")
        
        dataset_filename = config['dataset_name']
        dataset_path = os.path.join(paths['data'], dataset_filename)
        
        # Get column name from config, with fallback to 'expression'
        column_name = config.get('dataset_column', 'expression')
        word_list = load_word_list(dataset_path, column_name)
        
        tasks = create_openai_tasks(word_list, config, experiment_name, experiment_path)
        
        batch_files = create_batch_files(tasks, experiment_name, experiment_path)
        
        if batch_files:
            print(f"Successfully processed experiment: {experiment_name}")
            return True
        else:
            print(f"No batch files created for experiment: {experiment_name}")
            return False
            
    except Exception as e:
        print(f"Error processing experiment {experiment_name}: {e}")
        return False


def show_failed_status(experiment_path: str):
    """Display status of failed experiments."""
    print(f"\n{'='*60}")
    print("FAILED EXPERIMENTS STATUS")
    print(f"{'='*60}")
    
    failed_experiments = load_failed_experiments(experiment_path)
    
    if not failed_experiments:
        print("No failed experiments found!")
    else:
        print(f"Found {len(failed_experiments)} failed experiment(s):")
        for i, experiment in enumerate(failed_experiments, 1):
            print(f"  {i}. {experiment}")
        
        print("\nTo retry failed experiments:")
        print("  • Individual: python prepare_experiment.py <experiment_name> <experiment_path>")
        print("  • All failed: python prepare_experiment.py failed <experiment_path>")
        print("  • All experiments: python prepare_experiment.py all <experiment_path>")
    
    print(f"{'='*60}")


def process_multiple_experiments(experiment_names: list, all_configs: dict, experiment_path: str):
    """Process multiple experiments and show summary."""
    successful_count = 0
    failed_count = 0
    
    for experiment_name in experiment_names:
        if experiment_name not in all_configs:
            print(f"Warning: Experiment '{experiment_name}' not found in config.yaml, skipping...")
            continue
        
        success = process_experiment(experiment_name, all_configs[experiment_name], experiment_path)
        update_failed_experiments(experiment_name, success, experiment_path)
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful experiments: {successful_count}")
    print(f"Failed experiments: {failed_count}")
    print(f"{'='*60}")


def main():
    """Main execution function."""
    if len(sys.argv) != 3:
        print("Usage: python prepare_experiment.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>")
        print("  EXPERIMENT_NAME: experiment name, 'all', 'failed', or 'status'")
        print("  EXPERIMENT_PATH: Path to experiment folder")
        sys.exit(1)
    
    experiment_arg = sys.argv[1]
    experiment_path = sys.argv[2]
    
    # Ensure experiment path exists and has required structure
    if not os.path.exists(experiment_path):
        print(f"Error: Experiment path does not exist: {experiment_path}")
        sys.exit(1)
    
    ensure_experiment_directories(experiment_path)
    
    if experiment_arg.lower() == "status":
        show_failed_status(experiment_path)
        return
    
    all_configs = load_all_configs(experiment_path)
    
    if experiment_arg.lower() == "failed":
        failed_experiments = load_failed_experiments(experiment_path)
        if not failed_experiments:
            print("No failed experiments found!")
            return
        
        print(f"Retrying {len(failed_experiments)} failed experiments...")
        process_multiple_experiments(failed_experiments, all_configs, experiment_path)
        
    elif experiment_arg.lower() == "all":
        failed_experiments = load_failed_experiments(experiment_path)
        remaining_experiments = [name for name in all_configs.keys() 
                               if name not in failed_experiments]
        execution_order = failed_experiments + remaining_experiments
        
        print(f"Total experiments to process: {len(execution_order)}\n")
        process_multiple_experiments(execution_order, all_configs, experiment_path)
        
    else:
        experiment_name = experiment_arg
        
        if experiment_name not in all_configs:
            print(f"Error: Experiment '{experiment_name}' not found in config.yaml")
            print(f"\nAvailable experiments:")
            for exp in all_configs.keys():
                print(f"  • {exp}")
            sys.exit(1)
        
        success = process_experiment(experiment_name, all_configs[experiment_name], experiment_path)
        update_failed_experiments(experiment_name, success, experiment_path)
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()