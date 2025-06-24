import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
import time
from langchain.chains import RetrievalQA
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)

# Import refactored modules
from document_loader import load_documents
from vector_store import setup_vectordb
from llm_setup import setup_llama_llm

# RESPONSE AND REQUEST MODEL
class ChatbotRequest(BaseModel):
    query: str

class ChatbotResponse(BaseModel):
    answer: str
    sources: List[str] = []
    response_time: float

# Global variables for persistent components
vectordb = None
qa_chain = None

# DEFINE QA CHAIN
async def get_qa_chain():
    global qa_chain, vectordb
    if qa_chain is not None:
        return qa_chain
    if vectordb is None:
        docs_dir = config.docs_dir
        if not os.path.exists(docs_dir):
            logger.error(f"Directory {docs_dir} not found")
            raise FileNotFoundError(f"Directory {docs_dir} not found")
        try:
            all_docs = await load_documents(docs_dir)
            vectordb = setup_vectordb(all_docs, docs_dir=docs_dir, save_path=config.vectorstore_path)
        except Exception as e:
            logger.error(f"Error loading documents or setting up vector DB: {str(e)}")
            raise
    try:
        llm = setup_llama_llm()
    except Exception as e:
        logger.error(f"Error setting up LLM: {str(e)}")
        raise
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        logger.error(f"Error initializing QA chain: {str(e)}")
        raise
    return qa_chain

# INITIALIZE FAST API
app = FastAPI(
    title="ERP Document Chatbot API",
    description="API for querying ERP documents using Gemini and vector database",
    version="1.0.0"
)

# ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STARTUP EVENT
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing chatbot components...")
        await get_qa_chain()
        logger.info("ERP Document Chatbot API is ready!")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")

# HEALTH CHECK ENDPOINT
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "ERP Document Chatbot API is running"}

# MAIN ENDPOINT FOR CHATBOT
@app.post("/chat", response_model=ChatbotResponse)
async def chat(request: ChatbotRequest):
    # Input validation
    if not isinstance(request.query, str) or not request.query.strip():
        logger.warning("Received invalid or empty query.")
        raise HTTPException(status_code=422, detail="Query must be a non-empty string.")
    try:
        start_time = time.time()
        qa = await get_qa_chain()
        response = qa.invoke({"query": request.query})
        answer = response.get("result", "No answer found")
        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"][:3]:
                source = doc.metadata.get("source", "Unknown source")
                sources.append(os.path.basename(source))
        response_time = time.time() - start_time
        logger.info(f"Query processed in {response_time:.2f}s. Query: {request.query}")
        return ChatbotResponse(
            answer=answer,
            sources=sources,
            response_time=response_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("fastAPI:app", host="0.0.0.0", port=8000, reload=True)