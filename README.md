# ERPNext Document Chatbot API

A FastAPI-based chatbot API for querying ERPNext-related documents using advanced language models (LLaMA 2 via HuggingFace) and vector search (FAISS). The system is designed to help users interactively retrieve information from ERP documentation, with support for easy extension and robust error handling.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Adding/Updating Documents](#addingupdating-documents)
- [Troubleshooting](#troubleshooting)
- [Extending the Project](#extending-the-project)
- [License](#license)

---

## Features

- **Document Ingestion:** Loads `.txt` files from the `erp_docs/` directory (supports subfolders).
- **Vector Search:** Uses FAISS and HuggingFace embeddings for efficient document retrieval.
- **LLM Integration:** Uses LLaMA 2 (via HuggingFace) for generating answers.
- **Source Attribution:** Returns the top sources used for each answer.
- **CORS Enabled:** Ready for frontend integration.
- **Health Check Endpoint:** For easy deployment monitoring.
- **Robust Error Handling:** Clear error messages for missing dependencies, environment variables, or files.

---

## Project Structure

```
ERPNext chatbot/
  ├── erp_docs/                # Directory containing ERP documentation (.txt files)
  │    ├── hr/
  │    ├── payroll/
  │    ├── procurement/
  │    └── vendor/
  ├── fastAPI.py               # Main FastAPI application
  ├── fastAPI.ipynb            # (Optional) Jupyter notebook for experimentation
  ├── requirements.txt         # Python dependencies
  └── README.md                # Project documentation
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "ERPNext chatbot"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root with your HuggingFace API token:
     ```
     HUGGINGFACE_API_TOKEN=your_huggingface_token_here
     ```

4. **Add your ERP documentation:**
   - Place `.txt` files in the `erp_docs/` directory (organize by module as needed).

5. **Run the API:**
   ```bash
   uvicorn fastAPI:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Environment Variables

- `HUGGINGFACE_API_TOKEN`: Required for accessing HuggingFace LLaMA 2 model endpoints.

---

## How It Works

1. **Startup:**
   - On launch, the API loads all `.txt` documents from `erp_docs/`, splits them into chunks, and creates (or loads) a FAISS vector store for fast retrieval.

2. **Query Handling:**
   - When a user sends a query to `/chat`, the system:
     - Retrieves the most relevant document chunks using vector search.
     - Passes them to the LLaMA 2 model for answer generation.
     - Returns the answer, top sources, and response time.

3. **Persistence:**
   - The vector store is saved locally for fast subsequent startups.

---

## API Endpoints

### `GET /health`

- **Description:** Health check endpoint.
- **Response:**
  ```json
  {
    "status": "healthy",
    "message": "ERP Document Chatbot API is running"
  }
  ```

### `POST /chat`

- **Description:** Main endpoint for querying the chatbot.
- **Request Body:**
  ```json
  {
    "query": "How do I apply for leave?"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "To apply for leave, ...",
    "sources": ["leave_application_guide.txt"],
    "response_time": 0.42
  }
  ```

---

## Adding/Updating Documents

- Place new or updated `.txt` files in the appropriate subdirectory under `erp_docs/`.
- On next API startup, the documents will be loaded and indexed automatically.

---

## Troubleshooting

- **Missing dependencies:**  
  If you see an error about missing packages, run:
  ```bash
  pip install -r requirements.txt
  ```

- **Missing HuggingFace token:**  
  Ensure your `.env` file contains a valid `HUGGINGFACE_API_TOKEN`.

- **No documents found:**  
  Make sure you have `.txt` files in the `erp_docs/` directory.

- **Model/embedding errors:**  
  Check your internet connection and HuggingFace API token validity.

---

## Extending the Project

- **Switching LLMs:**  
  The code is modular—swap out `setup_llama_llm()` for another LLM setup (e.g., Gemini, OpenAI).

- **Adding new endpoints:**  
  Use FastAPI's standard route decorators.

- **Customizing retrieval:**  
  Adjust the `search_kwargs` or chunking strategy in `setup_vectordb()`.

- **Frontend integration:**  
  CORS is enabled for all origins; connect your frontend app directly to the API.

---

## License

Specify your license here (e.g., MIT, Apache 2.0).

---

**Maintainer:**  
Frank Lembalemba
