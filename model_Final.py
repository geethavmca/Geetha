import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter


os.environ["OPENAI_API_KEY"] = "{YOURAPIKEY}"

# Step 1: Convert PDF to text
floader = PyPDFLoader("examples.pdf")
documents = floader.load()

# Combine all page contents
full_text = "\n".join([doc.page_content for doc in documents])

# Save to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(full_text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)
chunks = text_splitter.create_documents([full_text])
# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0]) 
# Quick data visualization to ensure chunking was successful

# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# Create a DataFrame from the token counts
df = pd.DataFrame({'Token Count': token_counts})

# Create a histogram of the token count distribution
df.hist(bins=40, )

# Show the plot
plt.show()

# Step 2: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

# Step 3: Search similar documents
query = input("Enter Question?")
docs = db.similarity_search(query)

# Step 4: Extract context text
context = " ".join([doc.page_content for doc in docs])

# Step 5: Use QA pipeline directly
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Step 6: Ask question
result = qa({
    "question": query,
    "context": context
})


