import os
import aiofiles
import asyncio
import logging
from langchain_community.document_loaders import TextLoader
from typing import List

logger = logging.getLogger(__name__)

class AsyncTextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    async def load(self):
        try:
            async with aiofiles.open(self.file_path, mode='r', encoding='utf-8') as f:
                text = await f.read()
            # Simulate the output format of TextLoader.load()
            return [{
                'page_content': text,
                'metadata': {'source': self.file_path}
            }]
        except Exception as e:
            logger.error(f"Error loading {self.file_path}: {str(e)}")
            return []

async def load_documents(docs_dir: str) -> List:
    """Asynchronously load documents from the specified directory"""
    all_docs = []
    tasks = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".txt"):
                loader = AsyncTextLoader(os.path.join(root, file))
                tasks.append(loader.load())
    if tasks:
        results = await asyncio.gather(*tasks)
        for doc_list in results:
            all_docs.extend(doc_list)
    if not all_docs:
        logger.warning(f"No documents found in {docs_dir}")
    return all_docs 