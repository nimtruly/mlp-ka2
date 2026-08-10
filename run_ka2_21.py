"""
run_ka2_21.py — Run all cells of ka2_21.ipynb and save outputs.
"""
import nbformat
from nbclient import NotebookClient

print("Reading ka2_21.ipynb...")
with open('ka2_21.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Replace any Colab Google Drive mounting or /content/ paths with local paths if needed
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Replace Google Drive mount code or path substitutions
        if 'drive.mount' in cell.source:
            cell.source = "# Colab Drive mount commented out"
        cell.source = cell.source.replace('\"/content/train.csv\"', '\"train.csv\"')
        cell.source = cell.source.replace('\"/content/test.csv\"', '\"test.csv\"')
        cell.source = cell.source.replace('\'/content/train.csv\'', '\'train.csv\'')
        cell.source = cell.source.replace('\'/content/test.csv\'', '\'test.csv\'')

client = NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': '.'}})

print("Executing ka2_21.ipynb cells...")
client.execute()

with open('ka2_21.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully and saved with outputs!")
