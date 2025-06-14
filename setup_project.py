import os

# Define the project structure
folder_structure = [
    "B5W3/",
    "B5W3/data/",
    "B5W3/notebooks/",
    "B5W3/src/",
    "B5W3/reports/",
    "B5W3/.github/workflows/"
]

# Create the directories
for folder in folder_structure:
    os.makedirs(folder, exist_ok=True)

# Create essential files
files_to_create = [
    "B5W3/README.md",
    "B5W3/requirements.txt",
    "B5W3/.gitignore",
    "B5W3/dvc.yaml"
]

for file in files_to_create:
    with open(file, 'w') as f:
        if file.endswith('README.md'):
            f.write("# Insurance Risk Analytics Project\n\n")
            f.write("This project aims to analyze historical insurance claim data.\n")
        elif file.endswith('requirements.txt'):
            f.write("pandas\nnumpy\nmatplotlib\nseaborn\ndvc\n")
        elif file.endswith('.gitignore'):
            f.write("__pycache__/\n*.pyc\n*.pkl\n")
        elif file.endswith('dvc.yaml'):
            f.write("stages:\n  prepare:\n    cmd: python src/preprocess.py\n")

print("Folder structure created successfully!")