# ERPNext Document Chatbot API

A FastAPI-based chatbot API for querying ERPNext-related documents using advanced language models (LLaMA 2 via HuggingFace, OpenAI, Gemini) and vector search (FAISS). The system is designed to help users interactively retrieve information from ERP documentation, with support for easy extension, robust error handling, and flexible configuration.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
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
- **Multi-LLM Integration:** Supports LLaMA 2 (HuggingFace), OpenAI (GPT-3.5/4), and Gemini (Google) for generating answers.
- **Source Attribution:** Returns the top sources used for each answer.
- **CORS Enabled:** Ready for frontend integration.
- **Health Check Endpoint:** For easy deployment monitoring.
- **Robust Error Handling:** Clear error messages for missing dependencies, environment variables, or files.
- **Flexible Configuration:** All settings are managed via `config.yaml` and `.env`.

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
  ├── config.yaml              # Main configuration file
  ├── config.py                # Config loader module
  ├── document_loader.py       # Async document loading logic
  ├── vector_store.py          # Vector DB logic
  ├── llm_setup.py             # LLM setup logic (multi-provider)
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

3. **Set up configuration:**
   - Edit `config.yaml` to select your LLM provider and set model parameters:
     ```yaml
     llm_provider: huggingface  # or "openai" or "gemini"
     llm:
       repo_id: HuggingFaceH4/zephyr-7b-beta
       temperature: 0.1
       max_length: 512
       openai_model: gpt-3.5-turbo
       gemini_model: gemini-pro
     ```
   - Place `.txt` files in the `erp_docs/` directory (organize by module as needed).

4. **Set up environment variables:**
   - Create a `.env` file in the project root with the required API keys for your chosen LLM:
     ```env
     # For HuggingFace
     HUGGINGFACE_API_TOKEN=your_huggingface_token_here
     # For OpenAI
     OPENAI_API_KEY=your_openai_api_key_here
     # For Gemini
     GOOGLE_API_KEY=your_google_api_key_here
     ```

5. **Run the API:**
   ```bash
   uvicorn fastAPI:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Configuration

- **config.yaml:** All non-secret settings (paths, model names, chunk sizes, LLM provider, etc.)
- **.env:** All secrets and API keys

**Example: Switching LLMs**
- To use OpenAI, set in `config.yaml`:
  ```yaml
  llm_provider: openai
  llm:
    openai_model: gpt-3.5-turbo
    temperature: 0.2
    max_length: 512
  ```
  And in `.env`:
  ```env
  OPENAI_API_KEY=your_openai_api_key_here
  ```
- To use Gemini, set in `config.yaml`:
  ```yaml
  llm_provider: gemini
  llm:
    gemini_model: gemini-pro
    temperature: 0.2
    max_length: 512
  ```
  And in `.env`:
  ```env
  GOOGLE_API_KEY=your_google_api_key_here
  ```

---

## How It Works

1. **Startup:**
   - On launch, the API loads all `.txt` documents from `erp_docs/`, splits them into chunks, and creates (or loads) a FAISS vector store for fast retrieval. The vector store is only rebuilt if documents have changed.

2. **Query Handling:**
   - When a user sends a query to `/chat`, the system:
     - Retrieves the most relevant document chunks using vector search.
     - Passes them to the selected LLM for answer generation.
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

- **Missing API keys:**  
  Ensure your `.env` file contains the correct API key for your selected LLM provider.

- **No documents found:**  
  Make sure you have `.txt` files in the `erp_docs/` directory.

- **Model/embedding errors:**  
  Check your internet connection and API key validity.

---

## Extending the Project

- **Switching LLMs:**  
  Change the `llm_provider` and relevant fields in `config.yaml` and `.env`.

- **Adding new endpoints:**  
  Use FastAPI's standard route decorators.

- **Customizing retrieval:**  
  Adjust the `search_kwargs` or chunking strategy in `vector_store.py`.

- **Frontend integration:**  
  CORS is enabled for all origins; connect your frontend app directly to the API.

---

## License

Specify your license here (e.g., MIT, Apache 2.0).

---

**Maintainer:**  
Frank Lembalemba
