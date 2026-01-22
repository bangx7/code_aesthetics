import os
import json


updated_data = []
for dir in os.listdir("designarena_data_gt"):
    for file in os.listdir(os.path.join("designarena_data_gt", dir)):
        if file.endswith(".txt"):
            file_path = os.path.join("designarena_data_gt", dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    updated_data.append(
                        {
                            "id": dir,
                            "prompt": last_line,
                            "category": lines[0].strip()[10:],
                        }
                    )
                    break

# Sort the updated data by id
updated_data.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else float('inf'))

# Write the sorted data back to the file
with open("benchmark_data/all_prompt_transformed.jsonl", 'w', encoding='utf-8') as f:
    for data in updated_data:
        f.write(json.dumps(data) + '\n')
