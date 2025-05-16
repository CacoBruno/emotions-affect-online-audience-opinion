
import spacy
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load Portuguese tokenizer
nlp_pt = spacy.load("pt_core_news_sm")
stop_words_pt = set(stopwords.words('portuguese'))



def calculate_you_pronoun_proportion(text):
    """
    Calculates the proportion of comments containing "you" pronouns (você/vocês).
    (Related to Thelwall et al. (2022): Identifying 'you' pronoun usage as an indicator of strong PSI.)
    """
    you_pronouns = ["você", "vocês", "teu", "tu", "seu", "sua", "seus", "suas"]
    doc = nlp_pt(text)
    words = [token.text.lower() for token in doc if token.is_alpha and token.text.lower()]
    total_comments = len(words)
    list_you_count = [word for word in words if word in you_pronouns]
    you_count = len(list_you_count)
    result = you_count / total_comments if total_comments > 0 else 0
    return result