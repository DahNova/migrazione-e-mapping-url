import concurrent.futures
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# Ensure consistent language detection
DetectorFactory.seed = 0

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Function to load tokenizer
def load_tokenizer(language):
    try:
        if language == 'en':
            return nltk.tokenize.PunktSentenceTokenizer()
        elif language == 'it':
            # Italian tokenizer logic (if different from English)
            return nltk.tokenize.PunktSentenceTokenizer()
        else:
            return nltk.tokenize.PunktSentenceTokenizer()  # Default tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer for {language}: {e}")
        return None

# NLP Preprocessing Functions
def preprocess_text(text, language):
    try:
        # Tokenization and stemming/lemmatization logic
        tokenizer = load_tokenizer(language)
        if tokenizer is None:
            raise ValueError(f"Tokenizer not found for language {language}")

        tokens = tokenizer.tokenize(text)
        if language == 'en':
            lemmatizer = WordNetLemmatizer()
            processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stopwords.words('english')]
        elif language == 'it':
            stemmer = SnowballStemmer('italian')
            processed_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stopwords.words('italian')]
        else:
            processed_tokens = tokens  # Default case

        return ' '.join(processed_tokens)
    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")
        return text

def get_language(text):
    try:
        detected_lang = detect(text)
        return 'it' if detected_lang == 'it' else 'en'
    except:
        return 'en'  # Default to English if detection fails
      
def get_content(url, retries=3, timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    for attempt in range(retries):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            # Extract the title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""

            # Extract the meta description
            meta_desc = ""
            meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_desc_tag and meta_desc_tag.get('content'):
                meta_desc = meta_desc_tag['content'].strip()

            # Extracting text from <p> and all <h> tags
            texts = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            content = " ".join(element.get_text(strip=True) for element in texts)

            # Language Detection and NLP Preprocessing
            language = get_language(content)
            processed_content = preprocess_text(content, language)

            full_content = f"Title: {title}\nMeta Description: {meta_desc}\nContent: {processed_content}"

            if not full_content.strip():
                logging.warning(f"No content found in URL: {url}")
            return (url, full_content)  # Return a tuple of URL and full content
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(5)
    return (url, "")  # Return URL with empty content in case of failure

def read_urls_from_file(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file]

# Read URLs from files
url_list_a = read_urls_from_file("fromUrls.txt")
url_list_b = read_urls_from_file("toUrls.txt")

# Use requests.Session for improved performance
with requests.Session() as session:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        content_list_a = list(executor.map(get_content, url_list_a))
        time.sleep(5)  # Wait for 5 seconds between batches of requests
        content_list_b = list(executor.map(get_content, url_list_b))

# Create content dictionaries from the tuples
content_dict_a = {url: content for url, content in content_list_a}
content_dict_b = {url: content for url, content in content_list_b}

def calculate_similarity(content_dict_a, content_dict_b):
    vectorizer = TfidfVectorizer()
    combined_contents = list(content_dict_a.values()) + list(content_dict_b.values())
    tfidf_matrix = vectorizer.fit_transform(combined_contents)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(content_dict_a)], tfidf_matrix[len(content_dict_a):])
    return similarity_matrix

# Use the enhanced similarity function
similarity_matrix = calculate_similarity(content_dict_a, content_dict_b)

# Process and output the results
# Create a DataFrame to store the results
results = pd.DataFrame(columns=['From URL', 'To URL', 'Similarity'])

# Updating the DataFrame append to concat
for i, url_a in enumerate(content_dict_a.keys()):
    for j, url_b in enumerate(content_dict_b.keys()):
        similarity = similarity_matrix[i, j]
        temp_df = pd.DataFrame({'From URL': [url_a], 'To URL': [url_b], 'Similarity': [similarity]})
        results = pd.concat([results, temp_df], ignore_index=True)

results['Similarity'] = pd.to_numeric(results['Similarity'])

# Group by 'From URL' and get top 3 similarities
numero_risultati_per_url = 3
top_results = results.groupby('From URL').apply(lambda x: x.nlargest(numero_risultati_per_url, 'Similarity')).reset_index(drop=True)

# Sort results by 'From URL' and 'Similarity' in descending order
top_results.sort_values(by=['From URL', 'Similarity'], ascending=[True, False], inplace=True)

# Before saving to CSV
logging.info(f"Top matches: {top_results.head()}")  # Print the top rows of the DataFrame

# Write to CSV file
top_results.to_csv("result.csv", index=False)

logging.info("Script completed successfully.")