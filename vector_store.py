import os
import hashlib
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Any
from config import config

logger = logging.getLogger(__name__)

HASH_FILE = config.hash_file


def compute_docs_hash(docs_dir: str) -> str:
    """Compute a hash of all .txt files' contents and relative paths in docs_dir."""
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(docs_dir):
        for file in sorted(files):
            if file.endswith(".txt"):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                hash_md5.update(rel_path.encode())
                with open(os.path.join(root, file), "rb") as f:
                    while chunk := f.read(8192):
                        hash_md5.update(chunk)
    return hash_md5.hexdigest()


def read_stored_hash() -> str:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    return ""

def write_stored_hash(hash_val: str):
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(hash_val)


def setup_vectordb(documents: List, docs_dir: str = None, save_path: str = None) -> Any:
    """Create or load vector database, only rebuild if documents have changed."""
    docs_dir = docs_dir or config.docs_dir
    save_path = save_path or config.vectorstore_path
    current_hash = compute_docs_hash(docs_dir)
    stored_hash = read_stored_hash()
    if os.path.exists(save_path) and current_hash == stored_hash:
        logger.info(f"Loading existing vector store from {save_path} (documents unchanged)")
        embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)
        return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
        # return FAISS.load_local(save_path, embedding_model)
    logger.info("Creating new vector store...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks")
    embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)
    vectordb = FAISS.from_documents(split_docs, embedding_model)
    logger.info(f"Saving vector store to {save_path}")
    vectordb.save_local(save_path)
    write_stored_hash(current_hash)
    return vectordb 