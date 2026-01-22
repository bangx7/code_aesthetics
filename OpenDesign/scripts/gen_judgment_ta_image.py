import json
import os
import re
import concurrent.futures
from util.arena_judge_prompt import judge_single
from util.config_loader import config
from tqdm import tqdm
import base64
import logging 
import time
from datetime import datetime

logging.basicConfig(filename='missing_images.log', level=logging.INFO)

def get_current_time_string():
    """Get current time string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

json_time = get_current_time_string()

error_count = {}

from util.utils import (
    load_questions,
    chat_completion_openai,
)

def encode_image(image_path):  
    if os.path.exists(image_path):  
        with open(image_path, "rb") as image_file:  
            result = base64.b64encode(image_file.read()).decode('utf-8')  
            return result
    else:  
        # Log the missing image path  
        logging.info(f"Image not found: {image_path}")  
        # print(f"Image not found: {image_path}", flush=True)
        # Default base64 string for a 1x1 pixel transparent PNG image  
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgAB/1uXzRkAAAAASUVORK5CYII="


# get answer from model
def get_answer(model, conv, temperature, max_tokens):
    cfg = config.static_judge_config
    api_key = cfg.get('api_key')
    base_url = cfg.get('base_url')
    api_dict = {
        "api_key": api_key,
        "base_url": base_url,
    }
    output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
    return output


def judgment(**args):
    """
    Score the image of the answer.
    """
    question = args["question"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]
    user_prompt = args["user_prompt"]
    system_prompt = args["system_prompt"]
    question_id = question["id"]
 
    base64_image = encode_image(args["answer"])
    template = user_prompt
    
    conv = [{"role": "system", "content": system_prompt}]

    assert "<topic>" in user_prompt
    assert "<user_instruction>" in user_prompt

    user_prompt = template.replace("<topic>", question["category"]).replace("<user_instruction>", question["prompt"])
    
    conv.append(
        {"role": "user", "content": [
            {"type": "text", "text": f"{user_prompt}\n The Image of the answer of the assistant you need to judge is: \n"},
            {
                "image": base64_image
            }
        ]}
    )


    judgment = get_answer(
            model,
            conv,
            configs["temperature"],
            configs["max_tokens"],
    )
    judgment = judgment.replace("```json", "").replace("```", "").strip()
    result = {}
    try:
        result = json.loads(judgment)
    except:
        error_count[model] += 1
        print(f"Error: JSON format error when converting judge model's response. Ignoring question_id: {question_id}")
        if judgment == "" or judgment == None:
            result["judgment"] = "<API-Error>"
        else:
            result["judgment"] = judgment
        
    result["question_id"] = question_id
    result["judge_model"] = model
    result["user_prompt"] = question["prompt"]
    result["topic"] = question["category"]
    result["evaluate_model"] = args["evaluate_model"]

    with open(output_file, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Load from unified config - use static_judge config for image-based aesthetics evaluation
    model_name = config.model_to_evaluate
    static_judge_cfg = config.static_judge_config
    judge_model = static_judge_cfg.get('model', 'gpt-4o')
    
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    images_path = f'{base_output_dir}/images'
    
    configs = {
        "bench_name": config.get('benchmark.bench_name', 'opendesign'),
        "judge_model": judge_model,
        "temperature": static_judge_cfg.get('temperature', 0.0),
        "max_tokens": static_judge_cfg.get('max_tokens', 4096),
        "model_list": [model_name],
    }
    
    print(f"Evaluating model: {model_name}")
    print(f"Using static judge model: {judge_model}")
    
    print(f'Judge model: {configs["judge_model"]}')
    
    question_file = config.get('benchmark.data_file', 'benchmark_data/all_prompt.jsonl')

    questions = load_questions(question_file)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]

    output_files = {}
    output_dir = f"{base_output_dir}/model_judgment/judge_by_{configs['judge_model']}_images"
    model_answers = {}
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model.split('/')[-1]}_{json_time}.jsonl",
        )
        model_answers[model] = os.path.join(f"{base_output_dir}/images", model.split('/')[-1])


    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # endpoint_info = endpoint_list[configs["judge_model"]]
    print(f"Benchmarking {','.join(models)}")

    # Count the total number of tasks
    total_tasks = len(models) * len(questions)
    print(f"Total tasks to process: {total_tasks}")

    # Initialize error_count for all models before concurrent execution
    for model in models:
        error_count[model] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        futures = []
        task_count = 0
        
        # Add overall progress bar
        with tqdm(total=total_tasks, desc="Submitting judgment tasks") as pbar:
            for model in models:
                # Remove this line since we initialize above
                # error_count[model] = 0
                count = 0
                for idx, question in enumerate(questions):
                    question_id = question["id"]

                    kwargs = {}
                    kwargs["evaluate_model"] = model
                    kwargs["question"] = question
                    kwargs["answer"] = os.path.join(model_answers[model], f"image_{question_id}_tmp.png")
                    kwargs["user_prompt"] = judge_single
                    kwargs["configs"] = configs
                    kwargs["output_file"] = output_files[model]

                    kwargs["judge_model"] = configs["judge_model"]
                    kwargs["system_prompt"] = "You are a highly-skilled and impartial AI evaluator."
                    future = executor.submit(judgment, **kwargs)
                    futures.append(future)
                    
                    task_count += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'Model': model.split('/')[-1][:20], 
                        'Question': f"{idx+1}/{len(questions)}"
                    })

                if count > 0:
                    print(f"{count} number of existing judgments")

                print(f"The model {model}'s result will be saved in {output_files[model]}")

        # Process completed tasks progress bar
        print(f"\nProcessing {len(futures)} judgment tasks...")
        completed_count = 0
        with tqdm(total=len(futures), desc="Completing judgment tasks") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    completed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'Completed': f"{completed_count}/{len(futures)}"})
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    pbar.update(1)
                # future.result()
                # completed_count += 1
                # pbar.update(1)
                # pbar.set_postfix({'Completed': f"{completed_count}/{len(futures)}"})
    
    # Sort all result files by question_id
    print("\nSorting result files by question_id...")
    for model in models:
        output_file = output_files[model]
        if os.path.exists(output_file):
            # Read all lines from the file
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse JSON objects and sort by question_id
            json_objects = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        json_obj = json.loads(line)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON line in {output_file}: {line[:100]}...")
                        continue
            
            # Sort by question_id (convert to int for proper numerical sorting)
            json_objects.sort(key=lambda x: int(x.get('question_id', 0)))
            
            # Write back to file
            with open(output_file, 'w', encoding='utf-8') as f:
                for json_obj in json_objects:
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            
            print(f"Sorted {len(json_objects)} entries in {output_file}")
        else:
            print(f"Warning: Output file {output_file} does not exist")
    
    print("\nAll result files have been sorted by question_id.\n")
    print(f"The error count of each model is: {error_count}\n")
    print("Congratulations! All benchmarking tasks have been completed. ")


