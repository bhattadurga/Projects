import csv
from google.colab import files

# Test cases
test_cases = [
    ("How can I strengthen my immune system naturally?", "https://www.healthline.com/nutrition/ways-to-boost-immune-system", 5),
    ("What is the best way to manage hypertension in elderly people?", "https://www.cdc.gov/nchs/fastats/hypertension.htm", 5),
    ("Can eating a high-protein diet help with muscle building?", "https://www.webmd.com/diet/obesity/ss/slideshow-protein", 5),
    ("What are the symptoms of vitamin D deficiency?", "https://www.healthline.com/nutrition/vitamin-d-deficiency-symptoms", 5),
    ("How does stress affect the immune system?", "https://www.psychologytoday.com/us/basics/stress", 4),
    ("What are some tips for improving mental health during quarantine?", "https://www.who.int/news-room/q-a-detail/mental-health-and-covid-19", 5),
    ("What are the benefits of a Mediterranean diet?", "https://www.medicalnewstoday.com/articles/322511", 5),
    ("How can I reduce my carbon footprint on a daily basis?", "https://www.nrdc.org/stories/ways-reduce-your-carbon-footprint", 4),
    ("What are the best ways to stay hydrated?", "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/expert-answers/healthy-drinks/faq-20441816", 5),
    ("What are the health benefits of drinking green tea?", "https://www.medicalnewstoday.com/articles/324677", 5),
    ("How do I improve my sleep quality?", "https://www.sleepfoundation.org/sleep-hygiene/healthy-sleep-tips", 5),
    ("What are some tips for working from home effectively?", "https://www.thebalancecareers.com/tips-for-working-from-home-4067852", 5),
    ("How does exercise help with mental health?", "https://www.nimh.nih.gov/health/topics/exercise-and-depression", 5),
    ("What are some common symptoms of anxiety?", "https://www.anxiety.org/symptoms-of-anxiety", 5),
    ("What are the current travel restrictions for US citizens visiting Italy in 2025?", "https://www.italymagazine.com/2025-covid-19-travel-restrictions-italy", 5)
]


# Function to generate the CSV
def generate_csv():
    with open("/content/validation_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["user_prompt", "url_to_check", "func_rating", "custom_rating"])

        # Writing the data from test cases into the CSV
        for prompt, url, custom_rating in test_cases:
            writer.writerow([prompt, url, 3, custom_rating])

    print("CSV file 'validation_results.csv' created successfully!")

# Run the function to generate the CSV
generate_csv()

# Automatically download the CSV file
files.download("/content/validation_results.csv")
