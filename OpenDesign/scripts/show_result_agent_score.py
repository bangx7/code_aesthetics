import os
import json
import glob
from util.config_loader import config


def main():
    model_name = config.model_to_evaluate.split("/")[-1]
    avg_score = summarize_agent_score(model_name)
    print(f"Average agent score for model {model_name}: {avg_score}")
    
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    output_dir = f"./{base_output_dir}/model_judgment/final_results_agent_score/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f"{output_dir}/{model_name}_agent_score.jsonl", "a") as f:
        f.write(json.dumps({"model_name": model_name, "avg_score": avg_score}) + "\n")


def summarize_agent_score(model_name):
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    pattern = os.path.join(f"./{base_output_dir}/agent_score/{model_name}", f"{model_name}_*.jsonl")
    matching_files = glob.glob(pattern)
    if not matching_files:
        print(f"No .jsonl files found for model: {model_name}")
        return
    latest_file = max(matching_files, key=os.path.getmtime)
    print(f"Using the most recent file: {os.path.basename(latest_file)}")
    with open(latest_file, "r") as f:
        data = [json.loads(line) for line in f]

    avg_score = sum([sum(item["score"]) for item in data]) / len(data)
    return avg_score


if __name__ == "__main__":
    main()