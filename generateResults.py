""" USAGE:
python generateResults.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>

EXPERIMENT_NAME:
- <experiment_name>: process specific experiment.
- all: process all experiments.
- failed: retry failed experiments.
- status: show failed experiments.

EXPERIMENT_PATH:
- Path to experiment folder containing config.yaml, data/, results/, outputs/, etc.
"""

import json
import os
import glob
import numpy as np
import re
import time
import pandas as pd
import sys

# Add utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_yaml, get_experiment_paths, ensure_experiment_directories

def extract_word_input(text):
    """Extract the expression from the prompt text.
    The expression comes after 'La expresión es: ' and ends with '.'
    """
    # Pattern: Extract text between "La expresión es: " and "."
    pattern = r'La expresión es:\s*(.+?)\.'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        expression = match.group(1).strip()
        # Remove any quotes that might be present
        expression = expression.strip('"\'""„')
        return expression
    
    # Debug: print the text to see what we're working with
    print(f"Could not extract expression from text (first 300 chars):")
    print(text[:300])
    return None

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calculate_weighted_sum_1_to_7(top_logprobs_list):
    """Calculate weighted sum for numbers 1-7 from top logprobs.
    Only considers the first occurrence of each number 1-7.
    """
    weighted_sum = 0
    seen_numbers = set()
    
    # Iterate over the top logprobs
    for top_logprob in top_logprobs_list:
        try:
            token_str = str(top_logprob['token']).strip()
            # Check if token is a single digit from 1-7
            if token_str.isdigit() and len(token_str) == 1:
                token_value = int(token_str)
                if 1 <= token_value <= 7 and token_value not in seen_numbers:
                    logprob_value = top_logprob['logprob']
                    probability = np.exp(float(logprob_value))
                    weighted_sum += token_value * probability
                    seen_numbers.add(token_value)
        except (ValueError, KeyError):
            continue
    
    return weighted_sum

def openAI_processing(results_content_file, batches_content_file):
    match_key = 'custom_id'
    lookup = {entry[match_key]: entry for entry in batches_content_file if match_key in entry}
    combined_data = []
    
    for entry in results_content_file:
        entry_result = {}
        weighted_sum = None
        logprob = None
        
        if match_key in entry and entry[match_key] in lookup:
            combined_entry = {**entry, **lookup[entry[match_key]]}
            custom_id = combined_entry['custom_id']
            
            word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
            
            # Skip if no word input found
            if not word_input:
                print(f"Skipping entry {custom_id}: no expression found")
                continue
            
            # Only consider the first token
            if len(combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"]) >= 1:
                first_token_logprobs = combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"][0]
                
                # Get top logprobs (should be up to 10)
                top_logprobs_list = first_token_logprobs.get('top_logprobs', [])
                
                # Calculate weighted sum for numbers 1-7
                weighted_sum = calculate_weighted_sum_1_to_7(top_logprobs_list)
                
                # Get logprob of the chosen token
                logprob = first_token_logprobs.get('logprob')
            
            feature_value = combined_entry['response']['body']['choices'][0]['message']['content']

            entry_result['expression'] = word_input
            entry_result['familiarity'] = feature_value

            if logprob is not None:
                entry_result['logprob'] = logprob
            if weighted_sum is not None:
                entry_result['weighted_sum'] = weighted_sum

            combined_data.append(entry_result)

    return combined_data

def add_failed_experiment(experiment_name, experiment_path):
    """Add a failed experiment to the failed experiments file."""
    paths = get_experiment_paths(experiment_path)
    failed_file = os.path.join(paths['failed_experiments'], "failed_generateResults.txt")
    
    # Check if file exists and read existing data
    failed_experiments = []
    if os.path.exists(failed_file):
        try:
            with open(failed_file, 'r', encoding='utf-8') as f:
                failed_experiments = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            failed_experiments = []
    
    # Add experiment if not already in the list
    if experiment_name not in failed_experiments:
        failed_experiments.append(experiment_name)
        
        # Write back to file
        with open(failed_file, 'w', encoding='utf-8') as f:
            for exp in failed_experiments:
                f.write(f"{exp}\n")
        
        print(f"Added failed experiment: {experiment_name}")

def remove_failed_experiment(experiment_name, experiment_path):
    """Remove a successfully processed experiment from the failed experiments file."""
    paths = get_experiment_paths(experiment_path)
    failed_file = os.path.join(paths['failed_experiments'], "failed_generateResults.txt")
    
    if not os.path.exists(failed_file):
        return
    
    try:
        # Read existing failed experiments
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_experiments = [line.strip() for line in f.readlines() if line.strip()]
        
        # Remove the experiment
        failed_experiments = [exp for exp in failed_experiments if exp != experiment_name]
        
        if not failed_experiments:
            # Delete file if no failed experiments remain
            os.remove(failed_file)
            print(f"No failed experiments remaining. Removed {failed_file}")
        else:
            # Write back remaining failed experiments
            with open(failed_file, 'w', encoding='utf-8') as f:
                for exp in failed_experiments:
                    f.write(f"{exp}\n")
            print(f"Removed successful experiment: {experiment_name}")
    except Exception as e:
        print(f"Error updating failed experiments file: {e}")

def get_failed_experiments(experiment_path):
    """Get list of failed experiments."""
    paths = get_experiment_paths(experiment_path)
    failed_file = os.path.join(paths['failed_experiments'], "failed_generateResults.txt")
    
    if not os.path.exists(failed_file):
        return []
    
    try:
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_experiments = [line.strip() for line in f.readlines() if line.strip()]
        return failed_experiments
    except Exception:
        return []

def show_failed_experiments_status(experiment_path):
    """Show status of failed experiments."""
    failed_experiments = get_failed_experiments(experiment_path)
    
    if not failed_experiments:
        print("No failed experiments found.")
    else:
        print(f"Number of failed experiments: {len(failed_experiments)}")
        print("Failed experiments:")
        for exp_name in failed_experiments:
            print(f"  - {exp_name}")

def get_all_experiments_from_config(experiment_path):
    """Get all experiment names from config.yaml regardless of file availability."""
    try:
        paths = get_experiment_paths(experiment_path)
        config = read_yaml(paths['config'])
        experiments = list(config.get("experiments", {}).keys())
        print(f"All experiments from config: {experiments}")
        return experiments
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
        return []

def process_single_experiment(experiment_name, experiment_path):
    """Process a single experiment and return success status."""
    paths = get_experiment_paths(experiment_path)
    
    # Check if this experiment is already in failed list to avoid duplication
    failed_experiments = get_failed_experiments(experiment_path)
    is_in_failed_list = experiment_name in failed_experiments
    
    try:
        # SELECT THE BATCH AND RESULT FILES OF THE EXPERIMENT #
        matches_batches = glob.glob(os.path.join(paths['batches'], f"*{experiment_name}*.jsonl"))
        matches_results = glob.glob(os.path.join(paths['results'], f"*{experiment_name}*.jsonl"))

        if not matches_batches:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_path)
            print(f"No batches file found for experiment name '{experiment_name}' in '{paths['batches']}' folder.")
            return False

        if not matches_results:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_path)
            print(f"No results file found for experiment name '{experiment_name}' in '{paths['results']}' folder.")
            return False

        batches_file = matches_batches[0]
        results_file = matches_results[0]

        # CREATE OUTPUTS FOLDER IF IT DOESN'T EXIST#
        outputs_dir = os.path.join(paths['outputs'], experiment_name)
        os.makedirs(outputs_dir, exist_ok=True)

        timestamp = int(time.time())
        output_file = os.path.join(outputs_dir, f'output_{experiment_name}_{timestamp}.xlsx')

        print(f"Processing batch file: {batches_file}")
        print(f"Processing results file: {results_file}")
        
        results_content_file = read_jsonl(results_file)
        batches_content_file = read_jsonl(batches_file)
        
        print(f"Total entries in results: {len(results_content_file)}")
        print(f"Total entries in batches: {len(batches_content_file)}")

        # Only process OpenAI format
        if 'custom_id' not in batches_content_file[0]:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_path)
            print("Expected OpenAI batch format with 'custom_id', cannot process results.")
            return False

        combined_data = openAI_processing(results_content_file, batches_content_file)

        if not combined_data:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_path)
            print("No valid data to write (all entries were skipped).")
            return False

        all_fieldnames = list(combined_data[0].keys())

        df = pd.DataFrame(combined_data)
        df.to_excel(output_file, index=False, columns=all_fieldnames)

        print(f"Combined data written to {output_file}")
        print(f"  Total valid entries: {len(combined_data)}")
        
        # Remove from failed experiments if it was there
        if is_in_failed_list:
            remove_failed_experiment(experiment_name, experiment_path)
        
        return True

    except Exception as e:
        if not is_in_failed_list:
            add_failed_experiment(experiment_name, experiment_path)
        print(f"Error processing experiment '{experiment_name}': {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generateResults.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>")
        print("  EXPERIMENT_NAME: experiment name, 'all', 'failed', or 'status'")
        print("  EXPERIMENT_PATH: Path to experiment folder")
        exit()

    EXPERIMENT_NAME = sys.argv[1]
    EXPERIMENT_PATH = sys.argv[2]
    
    # Ensure experiment directories exist
    ensure_experiment_directories(EXPERIMENT_PATH)
    
    # Handle special commands
    if EXPERIMENT_NAME == "status":
        show_failed_experiments_status(EXPERIMENT_PATH)
        exit()
    
    if EXPERIMENT_NAME == "failed":
        failed_experiments = get_failed_experiments(EXPERIMENT_PATH)
        if not failed_experiments:
            print("No failed experiments to retry.")
            exit()
        
        print(f"Retrying {len(failed_experiments)} failed experiments...")
        for exp_name in failed_experiments:
            print(f"\nRetrying experiment: {exp_name}")
            process_single_experiment(exp_name, EXPERIMENT_PATH)
        exit()
    
    if EXPERIMENT_NAME == "all":
        # Get ALL experiments from config.yaml
        all_experiments_from_config = get_all_experiments_from_config(EXPERIMENT_PATH)
        
        if not all_experiments_from_config:
            print("No experiments found in config.yaml.")
            exit()
        
        print(f"Checking {len(all_experiments_from_config)} experiments from config...")
        
        # Filter experiments that have both batches and results files
        paths = get_experiment_paths(EXPERIMENT_PATH)
        experiments_to_process = []
        incomplete_experiments = 0
        
        for exp_name in all_experiments_from_config:
            matches_batches = glob.glob(os.path.join(paths['batches'], f"*{exp_name}*.jsonl"))
            matches_results = glob.glob(os.path.join(paths['results'], f"*{exp_name}*.jsonl"))
            
            has_batches = len(matches_batches) > 0
            has_results = len(matches_results) > 0
            
            if has_batches and has_results:
                experiments_to_process.append(exp_name)
                print(f"{exp_name}: has both batches and results files")
            elif has_batches or has_results:
                # Has only one of them, add to failed experiments
                failed_experiments = get_failed_experiments(EXPERIMENT_PATH)
                if exp_name not in failed_experiments:
                    add_failed_experiment(exp_name, EXPERIMENT_PATH)
                    incomplete_experiments += 1
                    print(f"{exp_name}: missing {'results' if has_batches else 'batches'} file, added to failed experiments")
                else:
                    print(f"{exp_name}: missing {'results' if has_batches else 'batches'} file, already in failed experiments")
            else:
                print(f"{exp_name}: no batches or results files found")
        
        if not experiments_to_process:
            print("No experiments with both batches and results files found.")
            exit()
        
        print(f"\nProcessing {len(experiments_to_process)} experiments with complete files...")
        successful = 0
        processing_failed = 0
        
        for exp_name in experiments_to_process:
            print(f"\nProcessing experiment: {exp_name}")
            if process_single_experiment(exp_name, EXPERIMENT_PATH):
                successful += 1
            else:
                processing_failed += 1
        
        total_failed = incomplete_experiments + processing_failed
        print(f"\nProcessing complete: {successful} successful, {total_failed} failed")
        if total_failed > 0:
            print(f"  - {incomplete_experiments} failed due to incomplete files")
            print(f"  - {processing_failed} failed during processing")
    else:
        # Process single experiment
        process_single_experiment(EXPERIMENT_NAME, EXPERIMENT_PATH)

