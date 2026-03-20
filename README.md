# Logit Lens/Tuned Lens
This project implements a logit lens and a tuned lens on pythia-14m, along with a user interface that allows easy comparison between the two. 
Either script can be run individually through the command line as well, with flags described below. Results are included at the bottom of this README.

## Files

- `lens.py`: terminal logit-lens view with Rich tables and progress output
- `tuned-lens.py`: tuned-lens training/inference script with save/load support
- `gui.py`: locally hosted html frontend for logit lens and tuned lens

## Running the code
Running `uv sync` will create a virtual environment with all the required dependencies. From there, you can run any of the three scripts on their own, with behavior described in more detail below.

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

## Results
The below images show the logit lens and tuned lens top token output on the input string "The trees are green and the sky is blue"

### Logit Lens
<img width="1313" height="476" alt="logit-lens" src="https://github.com/user-attachments/assets/b1c4f7d4-bb37-4256-a809-fee222147537" />

### Tuned Lens
<img width="1305" height="475" alt="tuned-lens" src="https://github.com/user-attachments/assets/063ab18a-2109-4a2c-bfe3-2721f0c0099f" />

## Reflection

The first thing that stood out is that, in the early layers, the tuned lens shows next-token predictions that are much more coherent than the standard logit lens. For example, at the token "is" in the input, the logit lens predicts the token "люч" at layer 0, while the tuned lens immidiately displays the token "a", which fits the local grammar much better and ultimately is the final prediction of the model.

Another pattern in the visualizations is the confidence trajectory. For the token after blue, the model reaches a reasonable prediction early, and confidence in that prediction generally rises across intermediate layers before dropping again near the final layer. This makes tuned-lens analysis useful for identifying where the model appears to be on track toward the final output, or at least where it has identified the right type of word to fill the next token slot.

Finally, the lens framework helps give some insight into which layers may be perfoming particularly important calculations to the final output. In the example above, you can look at the prediction after the token "green". At layer 3 it jumps from being mostly nonsensical to the final prediction, which is exactly correct relative to the original input. Though an interesting result, it is difficult to jump to conclusions about causality using this. Most of the information may have been gleaned from a previous layer in a non-obvious manner, and the lens framework doesn't have any way to know that.

At the same time, a major limitation is that single-token outputs do not reveal a full thought process. Whether the top prediction is "a" or "люч", a top token alone carries limited information about the model’s internal computation. This approach of just looking at the top token hides competing alternatives and uncertainty in the full distribution, but beyond that there are so many pieces that go into predicting the next token that an intermediate "thought" can't be represented by a single word. Intermediate reasoning thoughts also don't necessarily actually map to words that can be expressed in a model's vocabulary. Some layers may have learned to identify specific features about the input that are important for later layers to use to generate a final output, but don't necessarily actually produce an output that makes sense to an observer.

One of the big limitations of this analysis is scale. This is a small tuned lens applied to a relatively small language model. But even at larger scales, the core methodological issue doesn't go away, stronger single-token predictions still do not provide a complete account of model reasoning.

## References
Dataset: https://huggingface.co/datasets/NeelNanda/pile-10k
