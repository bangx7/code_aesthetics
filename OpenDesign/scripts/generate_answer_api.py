import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
from tqdm import tqdm

from util.azure_gpt4o import Openai, _get_api_infos_from_config
from util.config_loader import config
from openai import OpenAI

arena_prompt_template = {
    "website": """
    You are an expert web developer and designer specializing in modern websites. Create a complete, working HTML page with embedded CSS and JavaScript if needed. Feel free to use lightweight libraries like Tailwind CSS to enhance the design as long as they can be rendered in an iframe.
        Requirements:
        1. Create a fully functional, modern, and responsive website design
        2. Use only HTML, CSS, and JavaScript, but feel free to use libraries like Tailwind CSS to make the design better. Libraries such as ThreeJS, react three fiber, drei, @react-three/postprocessing, @react-three/cannon, d3, and recharts as additional libraries can be imported.
        3. Include all styles inline within <style> tags
        4. Focus on clean layouts, typography, and user experience
        5. Implement modern web design trends (gradients, shadows, smooth animations)
        6. Ensure excellent mobile responsiveness
        7. Include interactive elements where appropriate
        8. Make it production-ready and professional
        9. You must include all relevant script tags for libraries to work properly.
        Return ONLY the complete HTML code, starting with <!DOCTYPE html> and ending with </html>. Do not include any explanations or markdown formatting.
""",
"gamedev": """
You are an expert game developer specializing in browser-based games. Create a complete, working HTML page with an interactive game using embedded CSS and JavaScript.
        Requirements:
        1. Create a fully functional, playable browser game
        2. Use HTML, CSS, and JavaScript, but feel free to use libraries Tailwind CSS to make the game better as long as they will render in iframe. Libraries such as ThreeJS, react three fiber, drei, @react-three/postprocessing, @react-three/cannon, d3, and recharts (and others) can be imported.
        3. Include all styles inline within <style> tags
        4. Implement game mechanics, controls, and scoring systems
        5. Include game states (start screen, gameplay, game over)
        6. Add visual feedback, animations, and sound effects using Web Audio API if needed
        7. Ensure responsive design that works on both desktop and mobile
        8. Make the game engaging and fun to play
        9. You must include all relevant script tags for libraries to work properly.
        Return ONLY the complete HTML code, starting with <!DOCTYPE html> and ending with </html>. Do not include any explanations or markdown formatting.
""",
"3D_design": """
You are an expert in 3D graphics and WebGL. Create a complete, working HTML page with 3D graphics and animations using embedded CSS and JavaScript.
        Requirements:
        1. Create a fully functional 3D scene or application
        2. Use only CSS, and vanilla JavaScript with WebGL or CSS 3D transforms. Feel free to use lightweight libraries like Three.js or Babylon.js to make the 3D design better as long as they can be rendered in an iframe. Libraries such as ThreeJS, react three fiber, drei, @react-three/postprocessing, @react-three/cannon, d3, and recharts (and others) can be imported.
        3. Include all styles inline within <style> tags
        4. Implement 3D models, animations, and interactive controls
        5. Add proper lighting, materials, and camera controls
        6. Include smooth animations and user interaction
        7. Ensure good performance and responsive design
        8. Make it visually impressive and production-ready
        9. You must include all relevant script tags for libraries to work properly.
        Return ONLY the complete HTML code, starting with <!DOCTYPE html> and ending with </html>. Do not include any explanations or markdown formatting.
""",
"datavis": """
You are an expert in data visualization and interactive charts. Create a complete, working HTML page with dynamic data visualization capabilities using embedded CSS and JavaScript. Feel free to use lightwight libraries (as long as it can be rendered in an iframe) such as Tailwind CSS.
  
        Requirements:
        1. Create a fully functional data visualization application with interactive charts and graphs
        2. Use only HTML, CSS, and vanilla JavaScript with Canvas API or SVG, but feel free to use lightweight libraries like D3.js or Chart.js to make the visualizations better as long as they can be rendered in an iframe. Libraries such as ThreeJS, react three fiber, drei, @react-three/postprocessing, @react-three/cannon, d3, and recharts (and others) can be imported.
        3. Include all styles inline within <style> tags
        4. Ensure responsive design that adapts to different screen sizes
        5. Your goal is to make the design of the data visualization top-notch.
        6. You must include all relevant script tags for libraries to work properly.

        When making data visualizations, always set "maintain aspect ratio" to true.
        
        Return ONLY the complete HTML code, starting with <!DOCTYPE html> and ending with </html>. Do not include any explanations or markdown formatting.
""",
"UI": """
You are a world-class UI/UX designer and frontend engineer with a sharp eye for aesthetics, accessibility, and modern interaction design. Your task is to generate complete, production-ready HTML pages showcasing **visually stunning, highly interactive, and responsive UI components** using only HTML, CSS, and JavaScript.

 **Guidelines:**
1. Deliver a single, fully functional UI component as a complete HTML page
2. Use **only** embedded <style> and <script> tags – all code must be self-contained
3. You may use:
   - Tailwind CSS (via CDN)
   - Lightweight icon libraries (e.g., Heroicons)
   - Three.js, react-three-fiber, drei, d3, recharts (for advanced visuals), and others if you import them properly
4. Ensure **fully responsive design**, supporting both desktop and mobile layouts
5. Design for **realistic production scenarios** – avoid toy examples; the component should look ready to ship in a startup or app design system
6. You must include all relevant script tags for libraries to work properly.

 **Design Requirements (unless the user specifies otherwise):**
- Contemporary visual style: Soft shadows, rounded corners, clean typography, subtle gradients
- State handling: Show all interactive states (hover, active, loading, disabled, success)
- Microinteractions: Include smooth transitions and animations for interactive elements
- Accessibility: Use semantic HTML and ARIA roles where appropriate
- Use thoughtful spacing, sizing, and visual hierarchy

 **Bonus:**
- Add delight through animations, hover effects, or clever color usage
- Incorporate a beautiful and functional layout structure, not just the component

 Final Output:
Return ONLY the full, standalone HTML code (starting with <!DOCTYPE html>) and nothing else. No explanations, no markdown formatting.

If the user specifies a particular style (e.g., glassmorphism, brutalism, Material Design), follow their style instructions instead of the default design preferences.

"""
}   


short_prompt_template = (
    """
Design a website under the topic <topic>. The detailed requirements are <requirements>. The website must be a single page with all content visible without scrolling (fits within the viewport). Provide only the complete HTML code with embedded CSS and JavaScript in a single file. Do not include any explanatory text or comments outside the code. Ensure the code is clean, well-formatted, and functional in any modern browser. The output should be only the HTML code, **NO OTHER ANY TEXT OR EXPLANATION**.
"""
)


def extract_html_code(text: str) -> str:
    """Extract the HTML code between ```html and ``` markers if they exist."""
    import re

    pattern = r"```html\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fenced code block is found, return the original response.
    return text.strip()


# Load configuration
model_name = config.model_to_evaluate
num_threads = config.num_threads

print(f"Using API model: {model_name}")
print(f"Parallel threads: {num_threads}")


cat_dict = {
    "Game Dev": "gamedev",
    "Website": "website",
    "3D Design":"3D_design",
    "UI Component":"UI",
    "Data Visualization":"datavis"
}

# Load benchmark data
data_file = config.get('benchmark.data_file', 'benchmark_data/all_prompt.jsonl')
questions = datasets.load_dataset("json", data_files=data_file)["train"]

def build_prompt(example):
    """Convert a raw example into a chat-message format accepted by the API."""
    prompt = example["prompt"]
    category = example.get("category", "")
    
    # Map category to template, default to website if not found
    template_key = cat_dict.get(category, "website")
    system_content = arena_prompt_template.get(template_key, arena_prompt_template["website"])
    
    example["messages"] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    return example


# Apply transformation lazily – avoids materialising a new dataset, keeps RAM low.
questions = questions.map(build_prompt)



# ----------------------------------------------------------------------------
# Generate answers in parallel
# ----------------------------------------------------------------------------

responses = [None] * len(questions)

# We wrap API call in a function so that it can be executed in a thread pool.

def call_api(index: int, messages):
    """
    Call OpenAI-compatible API and return the raw text response.
    
    Supports multiple API providers based on model name.
    API configuration is loaded from config.yaml.
    """
    try:
        # Get OpenAI config from config.yaml
        openai_cfg = config.openai_config
        api_key = openai_cfg.get('api_key')
        base_url = openai_cfg.get('base_url')
        max_tokens = openai_cfg.get('max_tokens', 8192)
        temperature = openai_cfg.get('temperature', 0.7)
        
        
        # Default: OpenAI or OpenAI-compatible API using config
        api_infos = [{
        "model": openai_cfg.get('model'),
        "api_key": openai_cfg.get('api_key'),
        "base_url": openai_cfg.get('base_url'),
    }]
        openai_client = Openai(apis=api_infos)
            
            # Check if it's a reasoning model
        if any(indicator in model_name.lower() for indicator in ['o1', 'o3', 'gpt-5', 'reasoning']):
                response = openai_client.gpt5_call(messages, max_tokens=20000)
        else:
            response = openai_client.get_response(messages, max_tokens=max_tokens, temperature=temperature)
            
        return index, response
    
    except Exception as e:
        print(f"Error calling API for index {index}: {e}")
        return index, ""



with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all tasks first
    future_to_index = {
        executor.submit(call_api, idx, example["messages"]): idx for idx, example in enumerate(questions)
    }

    # Iterate over completed futures with a progress bar
    for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
        try:
            returned_idx, resp_text = future.result()
            resp_text = resp_text.replace("```html", "").replace("```", "").strip()
        except Exception as exc:
            returned_idx = future_to_index[future]
            print(f"[Error] index {returned_idx} generated an exception: {exc}")
            resp_text = ""
        responses[returned_idx] = resp_text

# ----------------------------------------------------------------------------
# Post-process responses and attach to dataset
# ----------------------------------------------------------------------------

def safe_token_len(text: str) -> int:
    """Approximate token length for the completion using whitespace split."""
    # Using whitespace split is a rough proxy that avoids external deps.
    return len(text.split())


processed_outputs = []
processed_token_lens = []
for resp in responses:
    try:
        html_only = extract_html_code(resp)
    except:
        html_only = "GPT4O_API_ERROR"
    processed_outputs.append(html_only)
    processed_token_lens.append(safe_token_len(html_only))

# Add new columns to the dataset
questions = questions.add_column("output", processed_outputs)
questions = questions.add_column("token_lens", processed_token_lens)

# ----------------------------------------------------------------------------
# Transform examples to the required JSONL format
# ----------------------------------------------------------------------------

def to_final_format(example):
    content = example["output"]
    if content.startswith("```html"):
        content = content.replace("```html", "").replace("```", "").strip()
    token_len = example["token_lens"]
    return {
        "question_id": example["id"],
        "answer_id": example["id"],
        "model_id": model_name,
        "choices": [
            {
                "index": 0,
                "turns": [{"content": content, "token_len": token_len}],
            }
        ],
    }


final_dataset = questions.map(to_final_format, remove_columns=questions.column_names)

# Save to disk
model_name_transformed = model_name.split("/")[-1]
base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
output_dir = f"./{base_output_dir}/model_answer/{model_name_transformed}"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{model_name_transformed}.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for example in final_dataset:
        json.dump(example, f, ensure_ascii=False)
        f.write("\n")

print(f"Data saved to {output_path}")
