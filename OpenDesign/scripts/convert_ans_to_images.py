import json
import os
from util.html2img import take_screenshot_dir_optimized
from tqdm import tqdm
from util.config_loader import config
import time
import re


def generate_image(model_name, model_answer_file, img_save_dir):
    model_answer_data = []
    with open(model_answer_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                model_answer_data.append(json.loads(line))
    img_save_dir = os.path.join(img_save_dir, model_name.split("/")[-1])
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    print(f"Generating temporary HTML files for {model_name}...")
    count=0
    for item in tqdm(model_answer_data):
        question_id = item["question_id"]  
        answers = item['choices'][0]['turns'][0]['content']
        answers = "<html><body><h1></h1></body></html>" if answers == "" or answers == None else answers
        if "```html" in answers:
            answers = answers.replace("```html", "").replace("```", "").strip()
        answers = re.sub(r"<think>.*?</think>", "", answers, flags=re.DOTALL)

        output_img = f"image_{question_id}.png"  
        temp_html_file = os.path.join(img_save_dir, f"{output_img[:-4]}_tmp.html")
        with open(temp_html_file, 'w', encoding='utf-8') as f:
            f.write(answers)
            count+=1
    print(f"Generated {count}/{len(model_answer_data)} HTML files")
    time.sleep(1)
    print(f"Rendering images for {model_name}...")
    max_workers = config.get('screenshot.max_workers', 16)
    timeout = config.get('screenshot.timeout', 120)
    batch_size = config.get('screenshot.batch_size', 100)
    take_screenshot_dir_optimized(img_save_dir, max_workers=max_workers, timeout=timeout, batch_size=batch_size)


def generate_answer_images(model_answer_file, model_name, img_save_dir=None):
    if img_save_dir is None:
        base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
        img_save_dir = f'./{base_output_dir}/images'
    
    generate_image(model_name, model_answer_file, img_save_dir)
    print("Image generation completed.")


if __name__ == "__main__":
    model_name = config.model_to_evaluate
    model_name_transformed = model_name.split("/")[-1]
    
    base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
    answer_dir = f'./{base_output_dir}/model_answer'
    model_answer_file = os.path.join(answer_dir, model_name_transformed, f'{model_name_transformed}.jsonl')
    generate_answer_images(model_answer_file, model_name)