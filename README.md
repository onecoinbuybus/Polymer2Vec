# Polymer2Vec
Embedding representation of polymer derived from PI1M dataset

## Installation Requirements

- Python 3.x
- pandas
- numpy
- rdkit
- LightGBM
- Optuna
- scikit-learn
- gensim
- mol2vec

### Model Training with `quick_qspr.py`

Use the following command to quickly train a QSPR model:

```bash
python quick_qspr.py -in your_data.csv -x smi -y tg -o your_model

```

### Command-Line Arguments

- `-in`: The name of the input CSV file.
    - Example: `train_test.csv`
  
- `-x`: The name of the SMILES column.
    - Example: `smi`

- `-y`: The name of the target column.
    - Example: `tg`

- `-o`: The name of trained model will be saved.
    - Example: `your_model`
