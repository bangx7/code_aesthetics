import json
import os
from util.webagent import judge_website_batch
from concurrent.futures import ThreadPoolExecutor, as_completed
from util.azure_gpt4o import _get_api_infos_from_config
from util.config_loader import config
import time
import random


# Get API configuration from config.yaml for interactive judge
API_INFOS = _get_api_infos_from_config('interactive')


def _get_optimal_api_index(batch_index, retry_count, num_apis):
    """
    Get optimal API index with intelligent load balancing.
    Uses randomization + round-robin to distribute load evenly.
    """
    # Base selection with offset for retry attempts
    base_index = (batch_index + retry_count * 3) % num_apis
    
    # Add some randomization to prevent synchronized requests
    random_offset = random.randint(0, min(3, num_apis - 1))
    
    return (base_index + random_offset) % num_apis


def _process_batch_solutions(args, model_name):
    """
    Process a batch of 3 solutions using judge_website_batch for HTML types.
    
    Args:
        args: Tuple of (batch_index, batch_solutions_data)
        batch_solutions_data: List of tuples (index, solution_str, extra_info)
    
    Returns:
        List of (index, score) tuples
    """
    batch_index, batch_solutions_data = args
    
    print(f"🚀 Processing batch {batch_index} with {len(batch_solutions_data)} solutions...")
    
    # Separate HTML and plot solutions
    html_solutions = []
    
    for idx, solution_str, extra_info in batch_solutions_data:
        if extra_info is None:
            raise ValueError(f"Extra info is None for solution {idx}! Please check the dataset. ")
        
        category = extra_info.get("category", "website")

        html_solutions.append((idx, solution_str, extra_info))
    
    results = []
    
    # Process HTML solutions using batch processing
    if html_solutions:
        print(f"📊 Processing {len(html_solutions)} HTML solutions in batch...")
        
        # Prepare data for batch processing
        html_list = []
        instructions = []

        solution_indices = []
        
        cat_dict = {"website": "General website", "datavis": "Data visualization", "3D_design": "3D design", "gamedev": "Game dev", "UI": "UI component"}
        
        for idx, solution_str, extra_info in html_solutions:

            solution_indices.append(idx)
            
            if True:
                html_list.append(solution_str)
                category = extra_info.get("category", "website")
                prompt = extra_info.get("prompt", "")
                tmp_prompt = f"The topic of the webpage is {category}. The webpage is designed based on the command: {prompt}."
                instructions.append(tmp_prompt)
            else:
                html_list.append("")  # Placeholder for failed execution
                instructions.append("")  # Placeholder
                print(f"❌ Execution failed for solution {idx+1}")
        
        # Batch judge websites
        if True:
            max_retries = 3
            retry_count = 0     
            batch_results = None
            
            while retry_count < max_retries and batch_results is None:
                try:
                    # Use intelligent load balancing to select API
                    api_index = _get_optimal_api_index(batch_index, retry_count, len(API_INFOS))
                    selected_api = API_INFOS[api_index]
                    print(f"🔄 Batch {batch_index} attempt {retry_count + 1} using API endpoint {api_index}")
                    
                    batch_results = judge_website_batch(
                        html_list=html_list,
                        instructions=instructions,
                        API_INFO=[selected_api],
                        timeout_seconds=600,  # Extended timeout to allow full evaluation
                        model_name=model_name
                    )
                    
                    # If successful, break out of retry loop
                    if batch_results is not None:
                        break
                        
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"❌ Batch {batch_index} attempt {retry_count + 1} failed with {error_type}: {e}")
                    retry_count += 1
                    
                    # Add exponential backoff delay
                    if retry_count < max_retries:
                        delay = min(2 ** retry_count, 10)  # Max 10 seconds
                        print(f"⏳ Waiting {delay}s before retry...")
                        time.sleep(delay)
                    
            # Process results if batch processing succeeded
            if batch_results is not None:
                # Calculate final scores
                for i, (idx, solution_str, extra_info) in enumerate(html_solutions):
                    if False:
                        results.append((idx, -1))
                    else:
                        agent_score = batch_results[i]

                        ###############################
                        # calculate the total reward. #
                        ###############################

                        reward_score = agent_score
                        print(f"🎆 Solution {idx+1} batch score: {reward_score}")
                        results.append((idx, reward_score))
            else:
                # print(f"❌ Batch {batch_index} failed after {max_retries} attempts, falling back to individual processing")
                # # Fallback to individual processing
                # for idx, solution_str, extra_info in html_solutions:
                #     try:
                #         score = calculate_single_score(solution_str, extra_info["prompt"], extra_info["category"], idx)
                #         results.append((idx, score))
                #     except Exception as individual_e:
                #         print(f"❌ Error processing solution {idx+1}: {individual_e}")
                #         results.append((idx, 0.0))
                raise Exception(f"Batch {batch_index} failed after {max_retries} attempts, falling back to individual processing")
        else:
            # All HTML solutions failed execution
            for idx, _, _ in html_solutions:
                results.append((idx, 0))
    
    
    print(f"✅ Completed batch {batch_index}")
    
    # Adaptive delay based on batch index to reduce API pressure
    # Later batches get longer delays to prevent accumulating rate limit issues
    batch_delay = min(2 + (batch_index // 20) * 0.5, 5)  # 2-5 seconds delay
    time.sleep(batch_delay)
    
    return results


def default_compute_score_batch(solution_strs,extra_infos, model_name):
    """
    Batch version of compute_score function using 48 threads with 3 solutions per thread.
    Uses judge_website_batch for efficient HTML evaluation.
    Uses ThreadPoolExecutor to avoid multiprocessing pickling issues.
    
    Args:
        solution_strs: List of solution strings to evaluate
        extra_infos: List of extra info dictionaries containing prompt and category
    
    Returns:
        List of reward scores
    """
    start_time = time.time()
    print(f"\n\n🚀 Processing batch of {len(solution_strs)} solutions with 64 threads (3 solutions per thread)...\n")
    
    # Prepare arguments for parallel processing - group into batches of 3
    batch_size = 3
    batch_args = []
    
    for i in range(0, len(solution_strs), batch_size):
        batch_solutions = []
        for j in range(i, min(i + batch_size, len(solution_strs))):
            batch_solutions.append((j, solution_strs[j], extra_infos[j]))
        batch_args.append((i // batch_size, batch_solutions))
    
    # Initialize results list with correct length
    scores = [[0.0]] * len(solution_strs)

    # Use ThreadPoolExecutor with 72 threads to avoid pickling issues
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submit all batch tasks
        future_to_batch = {
            executor.submit(_process_batch_solutions, args, model_name): args[0] 
            for args in batch_args
        }
        
        # Collect results as they complete
        completed_batches = 0
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                for index, score in batch_results:
                    scores[index] = score
                completed_batches += 1
                print(f"📊 Progress: {completed_batches}/{len(batch_args)} batches completed")
            except Exception as e:
                batch_index = future_to_batch[future]
                print(f"❌ Error processing batch {batch_index}: {e}")
                # Set default scores for failed batch
                start_idx = batch_index * batch_size
                for j in range(start_idx, min(start_idx + batch_size, len(solution_strs))):
                    scores[j] = [0.0]
    
    print(f"🎉 Batch processing completed with 64 threads! Total solutions: {len(solution_strs)}\n")

    print(f"\n\n🕒 Total time taken: {((time.time() - start_time)/60):.2f} minutes")
    


    
    return scores

def main():
    model_name = config.model_to_evaluate
    model_name_short = model_name.split("/")[-1]
    
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    data_file = config.get('benchmark.data_file', 'benchmark_data/all_prompt.jsonl')
    
    with open(f"{base_output_dir}/model_answer/{model_name_short}/{model_name_short}.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} solutions for model {model_name}")
    
    with open(data_file, "r") as f:
        all_prompt = [json.loads(line) for line in f]
    extra_infos = []
    htmls = []
    for idx in range(len(data)):
        extra_infos.append({"prompt": all_prompt[int(data[idx]["question_id"])-1]["prompt"], "category": all_prompt[int(data[idx]["question_id"])-1]["category"]})
        htmls.append(data[idx]['choices'][0]['turns'][0]['content'])


    scores = default_compute_score_batch(htmls, extra_infos, model_name_short)
    print(f"\n🎆 Judge done!")
    
    timestamp = int(time.time())
    output_dir = f"{base_output_dir}/agent_score/{model_name_short}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/{model_name_short}_{timestamp}.jsonl"
    with open(output_file, "w") as f:
        for idx in range(len(scores)):
            f.write(json.dumps({"question_id": data[idx]["question_id"], "score": scores[idx]}) + "\n")
    
    print(f"Agent scores saved to {output_file}")

if __name__ == "__main__":
    main()

