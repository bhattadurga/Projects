import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import spacy
from textblob import TextBlob

# Load SpaCy's pre-trained NLP model
nlp = spacy.load("en_core_web_sm")

def rate_url_validity(user_query: str, url: str) -> dict:
    """
    Evaluates the credibility of a given URL by computing various metrics including
    author credentials, content depth, external references, social media sentiment, 
    source popularity, technical accuracy, and additional features like NER and readability.

    Args:
        user_query (str): The user's original query.
        url (str): The URL to analyze.

    Returns:
        dict: A dictionary containing scores for different credibility attributes.
    """

    # === Step 1: Fetch Page Content ===
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = " ".join([p.text for p in soup.find_all("p")])  # Extract paragraph text
    except Exception as e:
        return {"error": f"Failed to fetch content: {str(e)}"}

    # === Step 2: Author Credentials ===
    author_score = check_author_credentials(url)

    # === Step 3: Content Depth (Semantic Similarity using Hugging Face) ===
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    similarity_score = util.pytorch_cos_sim(model.encode(user_query), model.encode(page_text)).item() * 100

    # === Step 4: External References (Citations in the Article) ===
    external_references_score = check_external_references(url)

    # === Step 5: Social Media Sentiment (NLP Sentiment Analysis) ===
    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_result = sentiment_pipeline(page_text[:512])[0]  # Process first 512 characters
    social_sentiment_score = 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    # === Step 6: Source Popularity (Alexa Ranking) ===
    source_popularity_score = check_source_popularity(url)

    # === Step 7: Technical Accuracy (Keyword Presence and Fact-Check) ===
    technical_accuracy_score = check_technical_accuracy(page_text)

    # === Step 8: Named Entity Recognition (NER) for Reputable Sources ===
    ner_score = check_named_entities(page_text)

    # === Step 9: Readability Score using TextBlob ===
    readability_score = check_readability_score(page_text)

    # === Step 10: Compute Final Credibility Score ===
    final_score = (
        (0.15 * author_score) +
        (0.2 * similarity_score) +
        (0.2 * external_references_score) +
        (0.1 * social_sentiment_score) +
        (0.1 * source_popularity_score) +
        (0.1 * technical_accuracy_score) +
        (0.05 * ner_score) +
        (0.1 * readability_score)
    )

    return {
        "Author Credentials Score": author_score,
        "Content Depth Score": similarity_score,
        "External References Score": external_references_score,
        "Social Sentiment Score": social_sentiment_score,
        "Source Popularity Score": source_popularity_score,
        "Technical Accuracy Score": technical_accuracy_score,
        "Named Entity Recognition Score": ner_score,
        "Readability Score": readability_score,
        "Final Credibility Score": final_score
    }


# === Helper Function: Author Credentials Check ===
def check_author_credentials(url: str) -> int:
    """
    Checks the credentials of the article's author based on the URL.
    Trusted authors may have credentials such as 'Dr.' or 'Prof.' and are matched against a predefined list.

    Prompts:
    - Are the authors recognized experts in the field? 
    - Do the authors have credible titles (e.g., Dr., Prof.)?

    Args:
        url (str): The URL of the article to check.

    Returns:
        int: A score (0-100) indicating the credibility based on author credentials.
    """
    trusted_authors = ["Dr. Jane Doe", "Prof. John Smith", "Dr. Alice Johnson"]
    if any(author in url for author in trusted_authors):
        return 90
    return 50

# === Helper Function: External References Check ===
def check_external_references(url: str) -> int:
    """
    Checks the number of external references (links) in the article.
    
    Prompts:
    - How many external sources are cited in the article?
    - Does the article link to credible, authoritative sources?

    Args:
        url (str): The URL of the article to check.

    Returns:
        int: A score (0-100) based on the number of external references.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        references = soup.find_all("a", href=True)
        if len(references) > 10:
            return 90
        elif len(references) > 5:
            return 70
        return 50
    except:
        return 40

# === Helper Function: Source Popularity Check (Alexa Ranking) ===
def check_source_popularity(url: str) -> int:
    """
    Checks the popularity of the source using a mock list (replace with actual Alexa API).

    Prompts:
    - Is the source of the article well-known and widely trusted?
    - Does the source have high traffic or a strong reputation?

    Args:
        url (str): The URL of the article to check.

    Returns:
        int: A score (0-100) based on the popularity and trustworthiness of the source.
    """
    popular_sources = ["bbc.com", "cnn.com", "nytimes.com"]
    for source in popular_sources:
        if source in url:
            return 90
    return 50

# === Helper Function: Technical Accuracy Check ===
def check_technical_accuracy(text: str) -> int:
    """
    Checks if the text contains technical accuracy based on the presence of important keywords.

    Prompts:
    - Does the article mention relevant technical terms like AI, Machine Learning, etc.?
    - Are these terms used correctly and in context?

    Args:
        text (str): The content of the article to analyze.

    Returns:
        int: A score (0-100) based on the technical accuracy of the content.
    """
    important_keywords = ["data science", "machine learning", "artificial intelligence", "deep learning"]
    technical_score = sum(1 for word in important_keywords if word.lower() in text.lower()) * 20
    return min(technical_score, 100)

# === Helper Function: Named Entity Recognition (NER) ===
def check_named_entities(text: str) -> int:
    """
    Perform Named Entity Recognition (NER) to check for reputable organizations or institutions.

    Prompts:
    - Does the article mention well-known and credible organizations like Harvard, MIT, etc.?
    - Are there any references to respected people or institutions?

    Args:
        text (str): The content of the article to analyze.

    Returns:
        int: A score (0-100) based on the identified named entities.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    trusted_entities = ["Harvard", "MIT", "Stanford", "Oxford"]
    matched_entities = sum(1 for entity in entities if entity in trusted_entities)
    return min(matched_entities * 30, 100)

# === Helper Function: Readability Score using TextBlob ===
def check_readability_score(text: str) -> int:
    """
    Calculate the readability score using TextBlob's polarity and subjectivity.

    Prompts:
    - How easy is it to read and understand the content of the article?
    - Is the tone clear, or does it become too complex for the reader?

    Args:
        text (str): The content of the article to analyze.

    Returns:
        int: A readability score (0-100).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 85  # Positive polarity, good readability
    elif polarity < 0:
        return 45  # Negative polarity, difficult readability
    return 65  # Neutral polarity, average readability
