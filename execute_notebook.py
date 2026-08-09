import nbformat
import sys
import io
import os
import contextlib

# Import nbclient
from nbclient import NotebookClient

with open('customer_churn_classification.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': '.'}})

print("Executing notebook cells...")
client.execute()

with open('customer_churn_classification.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully and saved with outputs!")
