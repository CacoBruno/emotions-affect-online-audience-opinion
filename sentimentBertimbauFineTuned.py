from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Load the tokenizer and the fine-tuned model
tokenizer = BertTokenizer.from_pretrained("./bertimbau-finetuned-sentiment")
model = BertForSequenceClassification.from_pretrained("./bertimbau-finetuned-sentiment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
print(device)

def sentimentPrediction(text):
    """
    Predicts sentiment and returns the class, probabilities, and confidence score.
    """
    # Preprocess the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Perform inference with the model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits and calculate probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).squeeze().tolist()
    
    # Determine the predicted class and confidence score
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_sentiment = ["Negative", "Positive"][predicted_class_idx]
    confidence_score = probabilities[predicted_class_idx]  # Probability of the predicted class

    # Return results
    return {
        "Sentiment": predicted_sentiment,
        "sentimentLabel": predicted_class_idx,
        "valence": round(probabilities[1], 2),
        "Probabilities": {
            "Negative": round(probabilities[0], 2),
            "Positive": round(probabilities[1], 2)
        },
        "Confidence Score": round(confidence_score * 100, 2)  # As a percentage
    }

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

# Custom dataset for DataLoader
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Custom dataset
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Optimized batch prediction function
def batch_sentiment_prediction(df, text_column, batch_size=64):  # Reduce batch_size if the error persists
    texts = df[text_column].tolist()
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    sentiments = ["Negative", "Positive"]
    results = []

    model.eval()
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Processing Batches"):
            # Batch preprocessing
            inputs = tokenizer(list(batch_texts), return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Inference
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            predicted_class_indices = torch.argmax(probabilities, dim=1)

            # Process results
            for i in range(len(batch_texts)):
                prob_list = probabilities[i].tolist()
                predicted_class_idx = predicted_class_indices[i].item()

                result = {
                    "Sentiment": sentiments[predicted_class_idx],
                    "sentimentLabel": predicted_class_idx,
                    "valence": round(prob_list[1], 2),
                    "Probabilities": {
                        "Negative": round(prob_list[0], 2),
                        "Positive": round(prob_list[1], 2)
                    },
                    "Confidence Score": round(prob_list[predicted_class_idx] * 100, 2)
                }
                results.append(result)

            # Free up GPU memory
            del inputs, outputs, logits, probabilities, predicted_class_indices
            torch.cuda.empty_cache()

    # Create the new column in the DataFrame
    df["sentimentDict"] = results
    return df
