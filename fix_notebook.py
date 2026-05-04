import json

file_path = "notebooks/colab_starter.ipynb"
with open(file_path, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if "--n-envs 16" in line:
                source[i] = line.replace("--n-envs 16", "--n-envs 8")
            if "--device cpu" in line:
                source[i] = line.replace("--device cpu", "--device auto")

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=2)
    f.write("\n")

print("Notebook fixed!")
