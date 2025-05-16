import json
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# Load the LIWC dictionary
with open("LIWC2007_Portugues_win.json", "r", encoding="utf-8") as f:
    liwc_data = json.load(f)

# Get the words categorized in LIWC
liwc_words = liwc_data["words"]

# Define categories based on JSON labels
categories = {
    "first_person": ["4", "5"],  # First-person singular pronouns
    "present": ["14"],  # Present focus
    "discrepancies": ["134"],  # Discrepancies
    "articles": ["10"],  # Articles (inverse scored)
}

def count_words(text, category_keys):
    """Count the number of words in the given text that belong to the specified category."""
    words = word_tokenize(text.lower())  # Tokenize the text
    count = sum(1 for word in words if any(label in category_keys for label in liwc_words.get(word, [])))
    return count


def calculate_liwc_verbal_immediacy(text):
    """Calculate the verbal immediacy score using LIWC-based categories."""
    words = word_tokenize(text.lower())  # Tokenize the text
    total_words = len(words) if words else 1  # Avoid division by zero
    
    # Calculate the presence of each category as a percentage of total words
    first_person_count = count_words(text, categories["first_person"])
    present_count = count_words(text, categories["present"])
    discrepancies_count = count_words(text, categories["discrepancies"])
    articles_count = count_words(text, categories["articles"])
    # Count words with more than 6 letters
    long_words_count = sum(1 for word in words if len(word) > 6)

    # Convert counts to percentages
    first_person_score = (first_person_count / total_words) * 100
    present_score = (present_count / total_words) * 100
    discrepancies_score = (discrepancies_count / total_words) * 100
    long_words_score = (long_words_count / total_words) * 100  # To be inversely scored
    articles_score = (articles_count / total_words) * 100  # To be inversely scored

    # Compute the verbal immediacy score as an arithmetic mean
    verbal_immediacy_score = (first_person_score + present_score + discrepancies_score - long_words_score - articles_score) / 5
    

    
    verbal_immediacy_normalized_score = (np.log1p(first_person_score) + np.log1p(present_score) + np.log1p(discrepancies_score) - np.log1p(long_words_score) - np.log1p(articles_score)) / 5


    return { 'comment' : text,
            'first_person_score' : {first_person_score, np.log1p(first_person_score)}, 
            'present_score' : {present_score, np.log1p(present_score)}, 
            'discrepancies_score' : {discrepancies_score, np.log1p(discrepancies_score)}, 
            'long_words_score' : {long_words_score, np.log1p(long_words_score)}, 
            'articles_score' : {articles_score, np.log1p(articles_score)}, 
            'verbal_immediacy_score' : verbal_immediacy_score,
            'verbal_immediacy_normalized_score' : verbal_immediacy_normalized_score}
