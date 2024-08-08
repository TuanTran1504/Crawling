import fitz  # PyMuPDF
import os
from pymongo import MongoClient
import ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
# MongoDB connection setup
client = MongoClient(
    "mongodb+srv://trandinhtuan1542:H5jBmdAtvakKvN8C@cluster0.eoqj8lz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Function to extract text from a text file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Function to split text into smaller chunks
def split_text(text, max_chunk_size=15000):
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


# Load documents from directory
loader = DirectoryLoader('./sample_file_launch', glob='*.*', show_progress=True)
data = loader.load()

# Process each document
for doc in data:
    file_path = doc.metadata['source']
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.pdf':
        document_content = extract_text_from_pdf(file_path)
    elif file_extension.lower() == '.txt':
        document_content = extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        continue

    #Split text
    splitter=RecursiveCharacterTextSplitter(chunk_size=30000,chunk_overlap=200)
    chunks = splitter.split_text(document_content)

    for chunk in chunks:
        # Get the embedding for the chunk
        response = ollama.embeddings(model='mxbai-embed-large', prompt=chunk)
        embedding = response["embedding"]

        # Create the document to insert into MongoDB
        document = {
            "text": chunk,
            "embedding": embedding  # Convert the embedding to a list for JSON serialization
        }

        # Insert the document into the collection
        collection.insert_one(document)
'''
{
        "mappings": {
            "dynamic": true,
            "fields": {
                "embedding": {
                    "dimensions": 1024,
                    "similarity": "cosine",
                    "type": "knnVector"
                }
            }
        }
}'''
