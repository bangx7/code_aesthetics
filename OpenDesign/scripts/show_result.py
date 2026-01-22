"""
Script to visualize the result of the benchmark.

1. Plot the result of each aspect of score. [user alignment, aesthetics and readability, structural integrity and responsiveness]

2. calculate the mean of each model. 

3. calculate the good rate (>=75) of each model. 

"""

import datetime
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from util.config_loader import config


def calculate_mean_and_good_rate(model_name):
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    # Use static_judge model for image-based aesthetics evaluation
    judge_model = config.static_judge_config.get('model', 'gpt-4o')
    base_dir = f"./{base_output_dir}/model_judgment/judge_by_{judge_model}_images"
    
    # Find the most recent .jsonl file for the given model
    pattern = os.path.join(base_dir, f"{model_name}_*.jsonl")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"No .jsonl files found for model: {model_name}")
        return
    
    # Sort files by modification time to get the most recent one
    latest_file = max(matching_files, key=os.path.getmtime)
    print(f"Using the most recent file: {os.path.basename(latest_file)}")
    
    # Read and process the JSONL file
    
    data = []
    total_score, align_score, aesth_score, read_score, good_count = 0, 0, 0, 0, 0
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    json_data = json.loads(line)
                    data.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue

    error_count = 0
    for item in data:
        try:
            total_score += int(item["total_score"])
            good_count += 1 if int(item["total_score"]) >= 75 else 0
            align_score += int(item["alignment_score"])
            aesth_score += int(item["aesthetics_score"])
            read_score += int(item["structure_score"])
        except:
            total_score += 0
            align_score += 0
            aesth_score += 0
            read_score += 0
            error_count += 1

    print(f"❗Warning: Error count(Judge model no response): {error_count}")

    zero_count = 0
    for item in data:
        if item["total_score"] == 0:
            zero_count += 1

    results = {}
    results[model_name] = {}

    results[model_name]["Total Score"] = round(total_score / (len(data) - error_count - zero_count), 2)

    results[model_name]["Alignment with User Instruction"] = round(align_score / (len(data) - error_count - zero_count), 2)

    results[model_name]["Aesthetics and Readability"] = round(aesth_score / (len(data) - error_count - zero_count), 2)

    results[model_name]["Structural Integrity and Responsiveness"] = round(read_score / (len(data) - error_count), 2)
    # Calculate good rate (percentage of scores >= 75)
    
    results[model_name]["Good rate"] = round(good_count / (len(data) - error_count - zero_count), 2) 

    return results, error_count



if __name__ == "__main__":
    model_name = config.model_to_evaluate.split("/")[-1]
    results, error_count = calculate_mean_and_good_rate(model_name)
    
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    save_dir = f"./{base_output_dir}/model_judgment/final_results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    final_results_file = os.path.join(save_dir, f'{model_name}_final_results.jsonl')
    results["model_name"] = model_name
    results["mode"] = "single"
    results["judge_model"] = config.static_judge_config.get('model', 'gpt-4o')
    results["Judge model no response"] = error_count
    results["judge_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(final_results_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")

    print(f"Final results saved as '{final_results_file}'")
