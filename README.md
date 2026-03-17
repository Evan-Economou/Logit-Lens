# Logit Lens/Tuned Lens
This project implements a logit lens and a tuned lens, along with a user interface that allows easy comparison between the two. 
Either script can be run individually through the command line as well, with flags described below.

## Files

- `lens.py`: terminal logit-lens view with Rich tables and progress output
- `tuned-lens.py`: tuned-lens training/inference script with save/load support
- `gui.py`: locally hosted html frontend for logit lens and tuned lens

## lens.py
`python lens.py`

### CLI Flags
- `text` (positional)
  - Default: `trees are green`
- `--top-k`
  - Type: integer
  - Default: `3`
  - Meaning: number of top tokens shown per cell

## tuned-lens.py
`python tuned-lens.py`

### CLI Flags
- `text` (positional)
  - Optional
  - Default: `trees are green`
- `--top-k`
  - Type: integer
  - Default: `3`
- `--train`
  - Flag (boolean)
  - If present: trains tuned-lens weights before analysis
- `--epochs`
  - Type: integer
  - Default: `3`
- `--max-samples`
  - Type: integer
  - Default: `60`
- `--parquet`
  - Type: path
  - Default: `train-00000-of-00001-4746b8785c874cc7.parquet` (next to the script)
- `--load`
  - Type: path
  - Default: `None`
  - Meaning: load tuned-lens weights from a .pt file
- `--save`
  - Type: path
  - Default: `tuned-lens-state.pt` (next to the script)
  - Meaning: save tuned-lens weights after training

## GUI
1. Start server with `python gui.py`
2. Then open `http://127.0.0.1:5000`

GUI supports:
- logit lens and tuned lens analysis
- training tuned lens from local parquet
- saving trained tuned-lens weights
- loading saved tuned-lens weights from a path

## Data and Pretrained Model
From the below Google Drive link you can download the training data used and a pretrained model  
https://drive.google.com/drive/folders/1MRWhiCejxr7cdBzD4wp23qsBrG3y7yk0?usp=drive_link

## References
Dataset: https://huggingface.co/datasets/NeelNanda/pile-10k
