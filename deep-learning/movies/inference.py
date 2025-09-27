import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from transformers import AutoTokenizer, TFBertForSequenceClassification
import optparse

# Load model weights and freezes layers to match the training model architecture
def load_model(model_path, model):
    def freeze_bottom_layers(model, num_layers_to_freeze):
        encoder = model.bert.encoder

        for layer in encoder.layer[:num_layers_to_freeze]:
            layer.trainable = False

    freeze_bottom_layers(model, num_layers_to_freeze=8)
    
    try:
        model.load_weights(model_path)
    except ValueError as e:
        print(f"Error loading weights: {e}")
    return model

# Predict a binary label - 1 for positive and 0 for negative
def predict(model, preprocessed_data):
    inputs = {k: v for k, v in preprocessed_data.items()}
    outputs = model(inputs)
    logits = outputs.logits
    predictions = tf.argmax(logits, axis=-1).numpy()
    return predictions

# Check if input is a file
def is_file(input_string):
    return os.path.isfile(input_string)

# Preprocess the input to be fed to model for inference
def prepare_data(input_data, tokenizer):
    
    if isinstance(input_data, list):
        texts = input_data
    elif is_file(input_data):
        if input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
            texts = df['text'].tolist()
        elif input_data.endswith('.txt'):
            with open(input_data, 'r') as file:
                texts = file.readlines()
                texts = [line.strip() for line in texts]
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")
    
    elif isinstance(input_data, str):
        texts = [input_data]
    else:
        raise ValueError("Unsupported input type. Please provide a file path, a single string, or a list of strings.")

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='tf'
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    return {'input_ids': input_ids, 'attention_mask': attention_masks}


def main():
    '''
    Run inference from the command line using the downloaded model weights on your file.
    '''
    parser = optparse.OptionParser()

    parser.add_option('-m', '--model', dest='model_filename', help='Path to the saved model file', metavar='FILE')
    parser.add_option('-i', '--input', dest='input_data', help='Input data file or string', metavar='INPUT')

    (options, args) = parser.parse_args()

    if not options.model_filename:
        parser.error('Model filename not given')
    if not options.input_data:
        parser.error('Input data not given')

    model_filename = options.model_filename
    input_data = options.input_data

    if not is_file(model_filename):
        parser.error(f'Model file {model_filename} does not exist')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model_4_bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    model = load_model(model_filename, model_4_bert)

    preprocessed_data = prepare_data(input_data, tokenizer)

    if isinstance(preprocessed_data, dict):
        predictions = predict(model, preprocessed_data)
        print(predictions)
    else:
        for batch in preprocessed_data:
            predictions = predict(model, batch)
            for prediction in predictions:
                print(prediction)

if __name__ == '__main__':
    main()