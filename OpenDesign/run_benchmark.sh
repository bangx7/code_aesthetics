#!/bin/bash
set -e

echo "=========================================="
echo "OpenDesign Benchmark"
echo "=========================================="

# Step 1: Generate answers
echo "[Step 1/5] Generating HTML outputs..."
# Check use_api setting in config.yaml to determine which script to run
USE_API=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['use_api'])")
if [ "$USE_API" = "True" ]; then
    echo "Using API mode..."
    python scripts/generate_answer_api.py
else
    echo "Using local inference mode..."
    python scripts/generate_answer.py
fi

# Step 2: Render HTML to images
echo ""
echo "[Step 2/5] Rendering HTML to images..."
python scripts/convert_ans_to_images.py

# Step 3: Judge static aesthetics
echo ""
echo "[Step 3/5] Judging static aesthetics..."
python scripts/gen_judgment_ta_image.py

# Step 4: Calculate interactive scores
echo ""
echo "[Step 4/5] Calculating interactive scores..."
python scripts/agent_score.py

# Step 5: Display results
echo ""
echo "[Step 5/5] Displaying results..."
python scripts/show_result.py
python scripts/show_result_agent_score.py

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
