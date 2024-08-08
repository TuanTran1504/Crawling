from langchain_core.vectorstores import VectorStore
from pymongo import MongoClient
import ssl
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param
from sentence_transformers import SentenceTransformer
import ollama
from langchain_community.llms import Ollama
from langchain.embeddings.base import Embeddings

# MongoDB connection setup
client = MongoClient("mongodb+srv://trandinhtuan1542:H5jBmdAtvakKvN8C@cluster0.eoqj8lz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]


# Define the embeddings class
class OllamaEmbeddings(Embeddings):
    def __init__(self):
        self.model = ollama

    def embed_query(self, text):
        response = self.model.embeddings(model='mxbai-embed-large', prompt=text)
        return response["embedding"]

    def embed_documents(self, documents):
        return [self.embed_query(doc) for doc in documents]

# Initialize the embeddings
embeddings = OllamaEmbeddings()

vectorStore=MongoDBAtlasVectorSearch(collection, embeddings)

def query_data(query):
    docs=vectorStore.similarity_search(query)
    as_output=docs[0].page_content
    llm=Ollama(model="llama3")
    retriever=vectorStore.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})
    qa=RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=retriever)
    retriever_output=qa.invoke(query)
    return as_output, retriever_output


with gr.Blocks(theme=Base(), title="QA App using Vector Search + RAG") as demo:
    gr.Markdown("""# QA App using Vector Search + RAG""")
    textbox=gr.Textbox(label="Enter your Question:")
    with gr.Row():
        button=gr.Button("Submit", variant="primary")
    with gr.Column():
        output1= gr.Textbox(lines=1, max_lines=10, label="Output with just Atlas Vector Search")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Output with just Atlas Vector Search to Langchain's")

    button.click(query_data,textbox, outputs=[output1,output2])
demo.launch()

