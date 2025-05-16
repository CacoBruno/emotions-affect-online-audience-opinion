import pickle
import pandas as pd

# Load lexicons
EmotionLookupTable = 'SentStrength_Data\EmotionLookupTable.txt'
BoosterWordList = 'SentStrength_Data\BoosterWordList.txt'
EmoticonLookupTable = 'SentStrength_Data\EmoticonLookupTable.txt'
SentimentLookupTable = 'SentStrength_Data\SentimentLookupTable.txt'

def load_lexicon(path):
    lexicon = pd.read_csv(path, sep="\t", names=["word", "sentiment"], encoding='utf-8')
    lexicon_dict = {row["word"]: row["sentiment"] for _, row in lexicon.iterrows()}
    return lexicon_dict

lexicon_1 = load_lexicon(EmotionLookupTable)
lexicon_2 = load_lexicon(BoosterWordList)
# lexicon_3 = load_lexicon(EmoticonLookupTable)
lexicon_4 = load_lexicon(SentimentLookupTable)

# Load trained lexicons from CSV files
lexicon_df = pd.read_csv('SentStrength_Data\lexiconTrained.csv')
lexicon_df = lexicon_df.dropna(subset=['Word'])

lexiconTrained = lexicon_df.set_index(['Word']).to_dict()['strengthValue']

lexicon_df_post = pd.read_csv('SentStrength_Data\lexiconTrainedPosts.csv')
lexicon_df_post = lexicon_df_post.dropna(subset=['Word'])

lexiconTrainedPost = lexicon_df_post.set_index(['Word']).to_dict()['strengthValue']

# Merge lexicons
lexicon_upload = lexiconTrained | lexicon_1 | lexicon_2 | lexicon_4
lexicon_upload_post = lexiconTrainedPost | lexicon_1 | lexicon_2 | lexicon_4

# Load saved tokenizer
with open("tokenizer.pkl", "rb") as handle:
    loaded_tokenizer = pickle.load(handle)
print("Tokenizer loaded successfully!")

with open("tokenizer_post.pkl", "rb") as handle:
    loaded_tokenizer_post = pickle.load(handle)
print("Tokenizer loaded successfully!")

from keras.models import load_model

# Load saved model
loaded_model = load_model("sentiment_strength_model.keras")
loaded_model_post = load_model("sentiment_strength_model_posts.keras")
print("Model .keras loaded successfully!")

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np

@tf.function
def predict_with_model(model, inputs, pos_tensor, neg_tensor):
    """
    Optimized function to make predictions with the model.
    """
    return model([inputs, pos_tensor, neg_tensor])

def sentimentStrength(text):
    """
    Predicts the sentiment strength of a text.

    Args:
        model: The trained model.
        text: The text to be analyzed.
        tokenizer: The tokenizer used during training.
        lexicon: The lexicon with sentiment scores.
        max_sequence_length: The maximum sequence length used during training.

    Returns:
        A dictionary with sentiment scores and predicted strength.
    """
    # Preprocess the text
    model = loaded_model
    tokenizer = loaded_tokenizer
    lexicon = lexicon_upload
    max_sequence_length = 100

    tokens = text.lower().split()
    pos_score = []
    neg_score = []

    for token in tokens:
        if token in lexicon:
            score = lexicon[token]
            if score > 0:
                pos_score.append(score)
            elif score < 0:
                neg_score.append(score)

    pos_avg = np.mean(pos_score) if pos_score else 0
    neg_avg = np.mean(neg_score) if neg_score else 0

    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Convert inputs to tensors
    inputs_tensor = tf.convert_to_tensor(padded_sequence, dtype=tf.float32)
    pos_tensor = tf.convert_to_tensor([[pos_avg]], dtype=tf.float32)
    neg_tensor = tf.convert_to_tensor([[neg_avg]], dtype=tf.float32)

    # Make prediction using the optimized function
    prediction = predict_with_model(model, inputs_tensor, pos_tensor, neg_tensor)

    # Return results
    return {
        "text": text,
        "strengthPredict": float(prediction[0][0])  # Convert to Python float
    }

def sentimentStrengthPost(text):
    """
    Predicts the sentiment strength of a post text.

    Args:
        model: The trained model.
        text: The text to be analyzed.
        tokenizer: The tokenizer used during training.
        lexicon: The lexicon with sentiment scores.
        max_sequence_length: The maximum sequence length used during training.

    Returns:
        A dictionary with sentiment scores and predicted strength.
    """
    # Preprocess the text
    model = loaded_model_post
    tokenizer = loaded_tokenizer_post
    lexicon = lexicon_upload_post
    max_sequence_length = 100

    tokens = text.lower().split()
    pos_score = []
    neg_score = []

    for token in tokens:
        if token in lexicon:
            score = lexicon[token]
            if score > 0:
                pos_score.append(score)
            elif score < 0:
                neg_score.append(score)

    pos_avg = np.mean(pos_score) if pos_score else 0
    neg_avg = np.mean(neg_score) if neg_score else 0

    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Convert inputs to tensors
    inputs_tensor = tf.convert_to_tensor(padded_sequence, dtype=tf.float32)
    pos_tensor = tf.convert_to_tensor([[pos_avg]], dtype=tf.float32)
    neg_tensor = tf.convert_to_tensor([[neg_avg]], dtype=tf.float32)

    # Make prediction using the optimized function
    prediction = predict_with_model(model, inputs_tensor, pos_tensor, neg_tensor)

    # Return results
    return {
        "text": text,
        "strengthPredict": float(prediction[0][0])  # Convert to Python float
    }
