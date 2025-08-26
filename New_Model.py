import requests
from bs4 import BeautifulSoup
import sentencepiece as spm
import os

# Step 1: Scrape website
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract only paragraph text
    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text() for p in paragraphs)
    return text

# Step 2: Save scraped text to a file
def save_text(text, filename="scraped.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

# Step 3: Train SentencePiece tokenizer
def train_tokenizer(input_file, model_prefix="tokenizer", vocab_size=8000):
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe"
    )

# Step 4: Load tokenizer and encode text
def encode_text(model_file, text):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    tokens = sp.encode(text, out_type=str)
    return tokens

# Main process
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
scraped_text = scrape_website(url)
save_text(scraped_text)

# Train tokenizer
train_tokenizer("scraped.txt")

# Encode sample text
tokens = encode_text("tokenizer.model", scraped_text[:1000])
print("Tokens:", tokens[:50])  # Print first 50 tokens
