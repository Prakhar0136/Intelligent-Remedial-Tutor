import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_pdf(pdf_path="data/sample.pdf", persist_directory="./chroma_db"):
    print(f"Loading PDF from {pdf_path}...")
    
    # 1. Load the PDF
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Please put a PDF in the 'data' folder.")
        return
        
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. Split the text into smaller, manageable chunks
    # We use chunks of 1000 characters with a 200 character overlap so we don't cut sentences in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split PDF into {len(chunks)} chunks.")
    
    # 3. Initialize our free, local Hugging Face embedding model
    print("Downloading/Loading Hugging Face Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Store the chunks in a local Chroma vector database
    print("Saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print("Ingestion complete! Your vector database is ready.")

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    # Run the ingestion
    ingest_pdf()