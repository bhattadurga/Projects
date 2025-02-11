###### run.py file ########
# Import the rate_url_validity function from the credibility_score module
from main import rate_url_validity

# Step 1: Define the user prompt and the URL to check
user_query = "What are the latest advancements in artificial intelligence and machine learning?"
url_to_check = "https://www.evonence.com/blog/top-5-breakthroughs-in-ai-and-machine-learning-for-2024"

# Step 2: Call the credibility evaluation function with the URL
result = rate_url_validity(user_query, url_to_check)

# Step 3: Display the evaluation result
print(result)
