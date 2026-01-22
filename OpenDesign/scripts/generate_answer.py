from vllm import LLM, SamplingParams
from tqdm import tqdm
import datasets
import json
import os
from util.config_loader import config

# Set CUDA devices from config
os.environ['CUDA_VISIBLE_DEVICES'] = config.get('local_inference.cuda_devices', '0')


# Load model configuration
model_name = config.get('local_inference.model_path')
if not model_name:
    model_name = config.model_to_evaluate

print(f"Benchmarking model: {model_name}")

prompt_template_arena_official = {
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

cat_dict = {"Website":"website", "3D Design":"3D_design", "Game Dev": "gamedev", "UI Component":"UI", "Data Visualization":"datavis"}

# Create an LLM
tensor_parallel = config.get('local_inference.tensor_parallel_size', 1)
max_model_len = config.get('local_inference.max_model_len', 10240)

if os.path.exists(model_name):
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel, trust_remote_code=True)
else:
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel, trust_remote_code=True, max_model_len=max_model_len)

print(f"model name: {model_name}")


gen_kwargs_vllm = {
    "max_tokens": config.get('generation.max_tokens', 10240),
    "temperature": config.get('generation.temperature', 0.7),
}
tokenizer = llm.get_tokenizer()



sampling_params = SamplingParams(**gen_kwargs_vllm)


data_file = config.get('benchmark.data_file', 'benchmark_data/all_prompt.jsonl')
questions = datasets.load_dataset('json', data_files=data_file)['train']

def convert_to_message(example):  
    """
    Need to revise regarding to actual opendesign data.
    """
    if example["category"] == "":
        messages = [{"role":"system", "content":prompt_template_arena_official["website"]},{"role": "user", "content": example["prompt"]}]  
    else:
        messages = [{"role":"system", "content":prompt_template_arena_official[cat_dict[example["category"]]]},{"role": "user", "content": example["prompt"]}]  
    example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(example["prompt"])
    return example  
questions = questions.map(convert_to_message)  


encoded_inputs = tokenizer.batch_encode_plus(  
    list(questions['prompt']),  
    add_special_tokens=False,
) 
input_ids = encoded_inputs['input_ids']  

outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
outputs_text = [x.outputs[0].text for x in outputs]

# Extract HTML code from outputs_text
def extract_html_code(text):
    """Extract HTML code between ```html and ``` markers"""
    import re
    pattern = r'```html\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return original text if no HTML code block found

# Apply HTML extraction to all outputs
outputs_text = [extract_html_code(text) for text in outputs_text]
token_lens = [len(x.outputs[0].token_ids) for x in outputs]

questions = questions.add_column("output", outputs_text)  
questions = questions.add_column("token_lens", token_lens) 

# Define transformation function  
def transform_example(example):  
    content = example['output']  
    # tokenized = tokenizer(content)  
    token_len = example['token_lens']   
      
    transformed_example = {  
        "question_id": example['id'],  
        "answer_id": example['id'],  
        "model_id": model_name,  
        "choices": [  
            {  
                "index": 0,  
                "turns": [  
                    {  
                        "content": content,  
                        "token_len": token_len  
                    }  
                ]  
            }  
        ],  
        # "tstamp": time.time()  
    }  
    return transformed_example  
  

transformed_dataset = questions.map(transform_example, remove_columns=questions.column_names)  

# Extract the last part after the final slash from model_name
model_name_transformed = model_name.split("/")[-1]


# Save as JSONL
base_output_dir = config.get('benchmark.output_dir', 'arena-bench-result')
output_dir = f'./{base_output_dir}/model_answer/{model_name_transformed}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f'{output_dir}/{model_name_transformed}.jsonl', 'w', encoding='utf-8') as f:  
    for example in transformed_dataset:  
        json.dump(example, f)  
        f.write('\n')  
  
print(f"Data saved to {output_dir}/{model_name_transformed}.jsonl")  
 