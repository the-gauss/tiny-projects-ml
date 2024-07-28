# BERT Inference Script

This repository contains a script to perform inference using a modified version of BERT model for sequence classification. The script can handle various input types, including CSV files, text files, single strings, and lists of strings (only supported in Jupyter Notebook).

## Requirements

- Python 3.6+
- TensorFlow 2.0+
- Transformers library by Hugging Face
- Pandas

You can install the required libraries using pip:

```sh
pip install tensorflow transformers pandas
```

## Model Weights

Download the model weights from the following link and place them in the project directory:
[Download Model Weights](https://drive.google.com/file/d/1oPnVUU-9DetlfbZcI79hIId23joXskkw/view?usp=drive_link)

## Usage

### Command-Line Interface

To run the script from the command line, you need to provide the path to the saved model weights and the input data. The input data can be a CSV file, a text file, or a single string.


#### CSV File

The CSV file should have a column named `text` containing the reviews.

```sh
python inference.py -m path/to/saved_weights.h5 -i path/to/input_file.csv
```

#### Text File

The text file should contain one review per line.

```sh
python inference.py -m path/to/saved_weights.h5 -i path/to/input_file.txt
```

#### Single String

You can also provide a single string directly.

```sh
python inference.py -m path/to/saved_weights.h5 -i "This is a single review."
```

Options ```-m``` and ```-i``` can be replaced with their long forms ```--model``` and ```--input``` respectively, as well.

### Script Overview

The `inference.py` script performs the following steps:

1. **Argument Parsing**: Parses command-line arguments for the model weights file and input data.
2. **Model Loading**: Loads the BERT model and weights.
3. **Data Preparation**: Prepares the input data for the model. Supports CSV files, text files, single strings, and lists of strings (only in Jupyter Notebooks).
4. **Prediction**: Runs the model on the preprocessed data and prints the predictions.

**The predictions will be a list of binary integer labels, 0 for negative and 1 for positive review. E.g. [1 1 0] represents first two reviews were positive while the last was negative.**

## Troubleshooting

If you encounter any issues, please ensure that:
- The input data file paths are correct.
- The model weights file path is correct.
- The required libraries are installed.