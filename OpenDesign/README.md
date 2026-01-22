# OpenDesign Benchmark


## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Benchmark Categories](#benchmark-categories)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Output Structure](#output-structure)
- [Advanced Usage](#advanced-usage)
- [Metrics](#metrics)
- [Troubleshooting](#troubleshooting)
- [Issues and Support](#issues-and-support)
- [Citation](#citation)

## ✨ Features

- **Multi-Category Evaluation**: Supports 5 design categories (Website, UI Component, 3D Design, Game Dev, Data Visualization)
- **Dual Evaluation Metrics**: 
  - Static aesthetics scoring using vision models
  - Interactive scoring using automated web agents
- **Flexible Inference Modes**: 
  - API-based inference (OpenAI, Claude, custom endpoints)
  - Local inference with vLLM
- **Automated Rendering**: Converts HTML to screenshots using Playwright
- **Parallel Processing**: Multi-threaded execution for faster benchmarking
- **Comprehensive Metrics**: Alignment, aesthetics, structural integrity, and interactive functionality scoring

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Modern web browser (for Playwright)
- CUDA-compatible GPU (optional, for local inference)

### Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright dependencies
playwright install
```

### Required Packages

```
pyyaml>=6.0
datasets>=2.14.0
tqdm>=4.65.0
playwright>=1.40.0
matplotlib>=3.7.0
numpy>=1.24.0
requests>=2.31.0
openai==1.1.1
selenium==4.15.2
pillow==10.1.0
vllm  # Only needed for local inference
```

## 🎯 Quick Start

### 1. Configure Your Benchmark

Copy an example configuration and modify it:

```bash
# For API-based evaluation
cp configs_example/openai_api.yaml config.yaml

# For local inference
cp configs_example/local_inference.yaml config.yaml
```

Edit `config.yaml` with your API keys and settings:

```yaml
model_to_evaluate: "gpt-4o"
use_api: true

openai:
  api_key: "your-api-key-here"
  base_url: null
  model: "gpt-4o"
```

### 2. Run the Benchmark

```bash
# First, make sure you are in the OpenDesign folder
bash run_benchmark.sh
```

This will execute the full pipeline:
1. Generate HTML outputs from prompts
2. Render HTML to images
3. Judge static aesthetics
4. Calculate interactive scores
5. Display final results

## ⚙️ Configuration

The benchmark uses a unified `config.yaml` file with the following sections:

### Model Configuration

```yaml
model_to_evaluate: "gpt-4o"  # Model name or path
use_api: true  # true for API mode, false for local inference
```

### API Configuration

```yaml
openai:
  api_key: YOUR_API_KEY
  base_url: null  # null for official OpenAI, or custom endpoint
  model: "gpt-4o"
  max_tokens: 8192
  temperature: 0.7
```

### Judge Models

```yaml
# Static aesthetics judge (vision-based)
static_judge:
  model: "gpt-4o"
  api_key: YOUR_API_KEY
  temperature: 0.0
  max_tokens: 4096

# Interactive functionality judge (agent-based)
interactive_judge:
  model: "gpt-4o"
  api_key: YOUR_API_KEY
  temperature: 0.0
  max_tokens: 4096
```

### Local Inference (vLLM)

```yaml
local_inference:
  model_path: "Qwen/Qwen2.5-Coder-32B-Instruct"
  tensor_parallel_size: 4  # Number of GPUs
  max_model_len: 10240
  cuda_devices: "0,1,2,3"
```

### Performance Tuning

```yaml
generation:
  num_threads: 32  # Parallel API calls

screenshot:
  max_workers: 16  # Parallel screenshot rendering
  timeout: 120
  batch_size: 100

agent_score:
  max_workers: 16  # Parallel agent evaluation
  batch_size: 3
  timeout_seconds: 600
```

### Example Configurations

See `configs_example/` for complete examples:
- `openai_api.yaml` - OpenAI API
- `claude_api.yaml` - Anthropic Claude API
- `local_inference.yaml` - Local model with vLLM
- `vllm_local.yaml` - vLLM server mode
- `qwen_dashscope.yaml` - Alibaba Cloud DashScope

## 🎨 Benchmark Categories

The benchmark includes **840 prompts** across 5 categories:

**General Website**: 60.9%

**Data Visualization**: 4.8%

**3D Design**: 14.6%

**Game Development**: 13.6%

**UI Component**: 4.9%


## 📊 Evaluation Pipeline

The benchmark follows a 5-step automated pipeline:

### Step 1: Generate HTML Outputs

```bash
# API mode
python scripts/generate_answer_api.py

# Local inference mode
python scripts/generate_answer.py
```

- Loads prompts from `benchmark_data/all_prompt.jsonl`
- Applies category-specific system prompts
- Generates HTML/CSS/JS code
- Saves to `arena-bench-result/model_answer/`

### Step 2: Render HTML to Images

```bash
python scripts/convert_ans_to_images.py
```

- Takes screenshots of generated HTML pages
- Uses Playwright for rendering
- Saves to `arena-bench-result/images/`

### Step 3: Judge Static Aesthetics

```bash
python scripts/gen_judgment_ta_image.py
```

- Vision model evaluates screenshots
- Scores on three dimensions:
  - **Alignment with User Instruction**
  - **Aesthetics and Readability**
  - **Structural Integrity and Responsiveness** 
- Saves to `arena-bench-result/model_judgment/`

### Step 4: Calculate Interactive Scores

```bash
python scripts/agent_score.py
```

- Web agent interacts with live HTML pages
- Tests functionality and interactive elements
- Evaluates user experience through automated actions
- Saves to `arena-bench-result/agent_score/`

### Step 5: Display Results

```bash
python scripts/show_result.py
python scripts/show_result_agent_score.py
```

- Aggregates all evaluation metrics
- Calculates mean scores and good rates (≥75)
- Saves final results to `arena-bench-result/model_judgment/final_results/`

## 📁 Output Structure

```
arena-bench-result/
├── model_answer/
│   └── {model_name}/
│       └── {model_name}.jsonl          # Generated HTML code
├── images/
│   └── {model_name}/
│       ├── image_1_tmp.png             # Rendered screenshots
│       └── ...
├── model_judgment/
│   ├── judge_by_{judge_model}_images/
│   │   └── {model_name}_{timestamp}.jsonl  # Static scores
│   └── final_results/
│       └── {model_name}/
│           └── {model_name}_final_results.jsonl
└── agent_score/
    └── {model_name}/
        └── {model_name}_{timestamp}.jsonl  # Interactive scores
```

## 🔧 Advanced Usage

### Run Individual Steps

```bash
# Only generate answers
python scripts/generate_answer_api.py

# Only render images (requires existing answers)
python scripts/convert_ans_to_images.py

# Only judge aesthetics (requires images)
python scripts/gen_judgment_ta_image.py

# Only calculate interactive scores (requires answers)
python scripts/agent_score.py

# View results
python scripts/show_result.py
```



### Use Custom Endpoints

```yaml
openai:
  base_url: "http://localhost:8000/v1"  # vLLM server
  api_key: "EMPTY"
```


## 📈 Metrics

### Static Aesthetics Scores

- **Total Score**: Average of three dimensions (0-100)
- **Alignment Score**: How well the design matches requirements
- **Aesthetics Score**: Visual appeal and design quality
- **Structure Score**: Code quality and responsiveness
- **Good Rate**: Percentage of outputs with Total Score ≥ 75

### Interactive Scores

- Automated agent interaction testing
- Functionality verification
- User experience evaluation
- Batch processing with retry logic




## 🐛 Troubleshooting

**Issue**: API rate limit errors
- Solution: Reduce `num_threads` in config or add delays

**Issue**: Playwright timeout errors
- Solution: Increase `screenshot.timeout` in config

**Issue**: vLLM OOM errors
- Solution: Reduce `max_model_len` or increase `tensor_parallel_size`

**Issue**: Missing screenshots
- Solution: Check `missing_images.log` for failed renders

## 💬 Issues and Support

If you encounter any problems or have questions about using this benchmark, please feel free to:

- Open an issue on GitHub
- Provide detailed information about your setup and error messages
- Check existing issues for similar problems and solutions

We welcome feedback and contributions to improve the benchmark!

## 🙏 Citation

If you find this codebase useful for your research, please use the following entry.

```bibtex
@misc{xiao2025codeaestheticsagenticreward,
      title={Code Aesthetics with Agentic Reward Feedback}, 
      author={Bang Xiao and Lingjie Jiang and Shaohan Huang and Tengchao Lv and Yupan Huang and Xun Wu and Lei Cui and Furu Wei},
      year={2025},
      eprint={2510.23272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.23272}, 
}
```

