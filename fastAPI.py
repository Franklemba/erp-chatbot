import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import time
from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel
import uvicorn

# ERROR HANDLING 

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please install required packages with: pip install langchain-google-genai google-generativeai langchain-huggingface faiss-cpu python-dotenv fastapi uvicorn")
    exit(1)
    

# INITIALIZE FAST API
app = FastAPI(
    title="ERP Document Chatbot API",
    description="API for querying ERP documents using Gemini and vector database",
    version="1.0.0"
)

# ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# RESPONSE AND REQUEST MODEL

class ChatbotRequest(BaseModel):
    query: str

# Define response model
class ChatbotResponse(BaseModel):
    answer: str
    sources: List[str] = []
    response_time: float

# Global variables for persistent components
vectordb = None
qa_chain = None


# LOAD DOCUMENT FUNCTION
def load_documents(docs_dir: str) -> List:
    """Load documents from the specified directory"""
    try:
        all_docs = []
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(".txt"):
                    try:
                        loader = TextLoader(os.path.join(root, file))
                        all_docs.extend(loader.load())
                        print(f"Loaded: {os.path.join(root, file)}")
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
        
        if not all_docs:
            print(f"Warning: No documents found in {docs_dir}")
            
        return all_docs
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        raise
    

# SETUP VECTOR DATABASE
def setup_vectordb(documents: List, save_path: str = "vectorstore") -> Any:
    """Create or load vector database"""
    try:
        # Check if we already have a saved vector store
        if os.path.exists(save_path):
            print(f"Loading existing vector store from {save_path}")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return FAISS.load_local(save_path, embedding_model)
        
        # Create new vector store
        print("Creating new vector store...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(split_docs, embedding_model)
        
        # Save for future use
        print(f"Saving vector store to {save_path}")
        vectordb.save_local(save_path)
        
        return vectordb
    except Exception as e:
        print(f"Error setting up vector database: {str(e)}")
        raise
    
# SETUP ILLAMA LLM
def setup_llama_llm():
    """Set up Meta's LLaMA 2 Chat model using Hugging Face"""
    try:
        print("Setting up LLaMA 2 model...")
        
        # Load environment variables
        load_dotenv()

        # Get Hugging Face token
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

        
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  
            huggingfacehub_api_token=hf_token,
            task="text-generation",
            temperature=0.1,
            model_kwargs={"max_length": 512}
        )
        return llm
    

    except Exception as e:
        print(f"Error setting up LLaMA 2 model: {str(e)}")
        raise

# DEFINE QA CHAIN
def get_qa_chain():
    """Initialize or return the QA chain"""
    global qa_chain, vectordb
    
    if qa_chain is not None:
        return qa_chain
    
    # Check if vectordb is already initialized
    if vectordb is None:
        # Load documents 
        docs_dir = "erp_docs"
        if not os.path.exists(docs_dir):
            raise FileNotFoundError(f"Directory {docs_dir} not found")
        
        all_docs = load_documents(docs_dir)
        vectordb = setup_vectordb(all_docs)
    
    # SETUP LLM
    # llm = setup_gemini_llm()
    llm = setup_llama_llm()
    
    # Create retriever and QA chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


# STARTUP EVENT
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        print("Initializing chatbot components...")
        get_qa_chain()
        print("ERP Document Chatbot API is ready!")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        # We'll continue without failing, and initialize when needed
        

# HEALTH CHECK ENDPOINT
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "ERP Document Chatbot API is running"}

# MAIN ENDPOINT FOR CHATBOT
@app.post("/chat", response_model=ChatbotResponse)
async def chat(request: ChatbotRequest):
    try:
        start_time = time.time()
        
        # Make sure QA chain is initialized
        qa = get_qa_chain()
        
        # Process the query
        response = qa.invoke({"query": request.query})
        answer = response.get("result", "No answer found")
        
        # Get sources
        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"][:3]:  # Top 3 sources
                source = doc.metadata.get("source", "Unknown source")
                sources.append(os.path.basename(source))
        
        # Calculate response time
        response_time = time.time() - start_time
        
        return ChatbotResponse(
            answer=answer,
            sources=sources,
            response_time=response_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# MAIN

if __name__ == "__main__":
    uvicorn.run("fastAPI:app", host="0.0.0.0", port=8000, reload=True)