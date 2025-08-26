import requests
import re
from bs4 import BeautifulSoup

#Scraping the Website, Clean and Write into a a Content.txt file.
url = "https://en.wikipedia.org/wiki/Natural_language_processing"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
paragraphs = soup.find_all('p')
text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
text = text.lower()                                # lowercase
text = re.sub(r'<.*?>', '', text)                  # remove HTML
text = re.sub(r'http\S+|www\S+', '', text)         # remove URLs
text = re.sub(r'[^a-z\s]', '', text)               # keep only letters
text = re.sub(r'\s+', ' ', text).strip()           # normalize spaces
with open("Content.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Scraping complete. Content saved to Content.txt.")



#Tokenizing and vocabularising the Scrapped text file

import sentencepiece as spm
# === Step 1: Preprocess and Clean the Text ===
with open("Content.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
# Clean lines: remove empty lines, trim, and truncate long lines
cleaned_lines = [line.strip()[:4000] for line in lines if line.strip()]
# Save cleaned text
with open("Cleaned_Content.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))
print("Number of cleaned lines:", len(cleaned_lines))
# === Step 2: Train SentencePiece Tokenizer ===
spm.SentencePieceTrainer.train(
    input='Cleaned_Content.txt',
    model_prefix='tokenizer',
    vocab_size=334
)
# === Step 3: Load the Trained Tokenizer ===
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
# === Step 4: Encode the Cleaned Text ===
# Combine cleaned lines into a single string
text = "\n".join(cleaned_lines)
# Encode into token IDs
encoded = sp.encode(text, out_type=int)
print("First 20 token IDs:", encoded[:20])
# === Step 5: Vocabulary Mapping (token → ID) ===
vocab_size = sp.get_piece_size()
vocab = {sp.id_to_piece(i): i for i in range(vocab_size)}
print("Sample vocab (first 10 tokens):", list(vocab.items())[:10])

