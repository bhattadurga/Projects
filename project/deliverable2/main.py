import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import spacy
import json

class URLValidator:
    """
    A URL validation class that evaluates the credibility of a webpage using multiple factors:
    domain trust, content relevance, fact-checking, bias detection, and citations.
    """

    def __init__(self):
        # Initialize external APIs and models
        self.serpapi_key = "7af58e07a928989508570e924fd67fdfc749c8ede38c05f2f265f33406d1e456"

        # Load models once to avoid redundant API calls
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        # Use a different model for fake news detection (DistilBERT for general text classification)
        self.fake_news_classifier = pipeline("text-classification", model="distilbert-base-uncased")

        # Using spaCy for Named Entity Recognition (NER) for authors and entities
        self.nlp = spacy.load("en_core_web_sm")

        # Sentiment analysis using a different model (RoBERTa for sentiment classification)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="roberta-base")

    def fetch_page_content(self, url: str) -> str:
        """ Fetches and extracts text content from the given URL. """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.text for p in soup.find_all("p")])  # Extract paragraph text
        except requests.RequestException:
            return ""  # Fail gracefully by returning an empty string

    def get_domain_trust(self, url: str, content: str) -> int:
        """ Computes the domain trust score based on available data sources. """
        trust_scores = []

        # Use fake news classifier for trust score
        if content:
            try:
                trust_scores.append(self.get_domain_trust_classifier(content))
            except Exception as e:
                print(f"Error classifying fake news: {e}")
                pass

        # Example: Add additional checks for SSL certificate verification
        if self.is_https(url):
            trust_scores.append(80)  # Higher score for HTTPS domains

        return int(sum(trust_scores) / len(trust_scores)) if trust_scores else 50

    def get_domain_trust_classifier(self, content: str) -> int:
        """ Uses a DistilBERT fake news detection model to assess credibility. """
        if not content:
            return 50  # Default score if no content available
        result = self.fake_news_classifier(content[:512])[0]  # Process only the first 512 characters
        return 100 if result["label"] == "LABEL_1" else 30 if result["label"] == "LABEL_0" else 50

    def is_https(self, url: str) -> bool:
        """ Checks if the URL uses HTTPS for security. """
        return url.lower().startswith("https://")

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        """ Computes semantic similarity between user query and page content. """
        if not content:
            return 0
        return int(util.pytorch_cos_sim(self.similarity_model.encode(user_query), self.similarity_model.encode(content)).item() * 100)

    def verify_facts(self, content: str) -> int:
        """ Cross-checks content with a fact-checking database. """
        if not content:
            return 60

        api_url = f"https://factcheckapi.com/v1/facts?query={content[:200]}"
        try:
            response = requests.get(api_url)
            data = response.json()
            return 90 if "facts" in data and data["facts"] else 50
        except Exception as e:
            return 60  # Default uncertainty score in case of error

    def check_scholar_references(self, url: str) -> int:
        """ Checks scholarly references using a research API. """
        api_key = self.serpapi_key  # Assuming the API key is stored in the class
        params = {"q": url, "engine": "scholar", "api_key": api_key}
        try:
            response = requests.get("https://researchapi.com/search", params=params)
            data = response.json()
            return min(len(data.get("results", [])) * 12, 100)  # Normalize based on results
        except Exception as e:
            return 0  # Default to no citations

    def detect_content_bias(self, content: str) -> int:
        """ Uses sentiment analysis to detect content bias. """
        if not content:
            return 60
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        return 90 if sentiment_result["label"] == "POSITIVE" else 60 if sentiment_result["label"] == "NEUTRAL" else 40

    def get_star_rating(self, score: float) -> tuple:
        """ Converts a score (0-100) into a 1-5 star rating. """
        stars = max(1, min(5, round(score / 20)))  # Normalize 100-scale to 5-star scale
        return stars, "⭐" * stars

    def generate_rating_explanation(self, reliability, relevance_score, fact_check_score, bias_score, citation_score, total_score) -> str:
        """ Generates a human-readable explanation for the score. """
        reasons = []
        if reliability < 60:
            reasons.append("The source has low reliability.")
        if relevance_score < 60:
            reasons.append("The content is not highly relevant to your query.")
        if fact_check_score < 60:
            reasons.append("Limited fact-checking information found.")
        if bias_score < 60:
            reasons.append("Potential bias detected in the content.")
        if citation_score < 40:
            reasons.append("Few scholarly references found for this content.")

        return " ".join(reasons) if reasons else "This source is highly reliable and relevant."

    def evaluate_content_quality(self, content: str) -> int:
        """ Evaluates the overall quality of the content (e.g., spelling, grammar, depth of information). """
        # A simple placeholder function to assess content quality
        if not content:
            return 50
        quality_score = 80  # Example of a predefined quality score
        if len(content.split()) < 300:  # Low word count = lower quality
            quality_score -= 20
        return quality_score

    def evaluate_trustworthiness(self, url: str) -> int:
        """ Assesses the trustworthiness of the domain based on known sources or reputation. """
        # Placeholder for actual trust assessment logic
        trusted_sources = ["nytimes.com", "bbc.com", "wikipedia.org"]
        domain = url.split("/")[2]
        if domain in trusted_sources:
            return 90
        return 50

    def get_star_description(self, stars: int) -> str:
        """ Converts the star rating into a more descriptive interpretation. """
        if stars == 5:
            return "Very Reliable"
        elif stars == 4:
            return "Reliable"
        elif stars == 3:
            return "Moderate"
        elif stars == 2:
            return "Low"
        else:
            return "Very Low"

    def analyze_sentiment(self, content: str) -> int:
        """ Analyzes sentiment (positive, neutral, or negative) of the content. """
        if not content:
            return 50  # Neutral if no content
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        if sentiment_result["label"] == "POSITIVE":
            return 90
        elif sentiment_result["label"] == "NEGATIVE":
            return 40
        return 60  # Neutral

    def evaluate_engagement(self, content: str) -> int:
        """ A simple heuristic to estimate how engaging the content is. """
        if not content:
            return 50
        # Engagement could be measured by factors like length, readability, etc.
        engagement_score = 60
        if len(content.split()) > 500:
            engagement_score += 10  # Longer content tends to be more engaging
        if "comments" in content or "feedback" in content:  # Check if there are interaction elements
            engagement_score += 10
        return engagement_score

    def rate_url_validity(self, user_prompt: str, url: str) -> dict:
        """ Main function to evaluate the validity of a webpage. """
        content = self.fetch_page_content(url)

        # Get all scores for the given URL and content based on various factors
        domain_trust = self.get_domain_trust(url, content)
        similarity_score = self.compute_similarity_score(user_prompt, content)
        fact_check_score = self.verify_facts(content)
        bias_score = self.detect_content_bias(content)
        citation_score = self.check_scholar_references(url)
        content_quality_score = self.evaluate_content_quality(content)
        trustworthiness_score = self.evaluate_trustworthiness(url)
        sentiment_score = self.analyze_sentiment(content)
        engagement_score = self.evaluate_engagement(content)

        # Final weighted score calculation
        final_score = (
            (0.2 * domain_trust) +
            (0.2 * similarity_score) +
            (0.2 * fact_check_score) +
            (0.1 * bias_score) +
            (0.1 * citation_score) +
            (0.1 * content_quality_score) +
            (0.05 * trustworthiness_score) +
            (0.05 * sentiment_score)
        )

        stars, icon = self.get_star_rating(final_score)
        explanation = self.generate_rating_explanation(
            domain_trust, similarity_score, fact_check_score, bias_score, citation_score, final_score
        )


        return {
            "raw_score": {                                        # The raw_score dictionary contains all the individual evaluation scores for the URL.
                "Domain Trust": domain_trust,                     # Domain Trust score assesses how trustworthy the domain is based on various factors like HTTPS usage and credibility.
                "Content Relevance": similarity_score,            # Content Relevance score measures how relevant the content of the webpage is to the user's query.
                "Fact-Check Score": fact_check_score,             # Fact-Check Score evaluates the credibility of the information based on fact-checking sources.
                "Bias Score": bias_score,                         # Bias Score checks if there is any detected bias in the content of the page (e.g., sentiment analysis).
                "Citation Score": citation_score,                 # Citation Score estimates how many scholarly references are available for the content.
                "Content Quality Score": content_quality_score,   # Content Quality Score evaluates factors like spelling, grammar, and depth of information provided on the page.
                "Trustworthiness Score": trustworthiness_score,   # Trustworthiness Score assesses the overall trustworthiness of the domain based on known sources and reputation.
                "Sentiment Score": sentiment_score,               # Sentiment Score analyzes the sentiment (positive, neutral, or negative) of the content.
                "Engagement Score": engagement_score,             # Engagement Score measures how engaging the content is based on factors like length, comments, and interaction.
                "Final Validity Score": final_score               # Final Validity Score is the weighted average of all the above scores to provide an overall trust rating.
            },
            "stars": {                                            # The stars dictionary contains information about the star rating.
                "score": stars,                                   # The numerical score represents the star rating (on a scale from 1 to 5).
                "icon": icon,                                     # The icon represents the visual star icon that corresponds to the score.
                "description": self.get_star_description(stars)   # The description provides a human-readable interpretation of the star rating (e.g., "Very Reliable").
            },
            "explanation": explanation                            # Explanation provides a summary of why the page received the given scores, outlining the strengths and weaknesses.
        }
