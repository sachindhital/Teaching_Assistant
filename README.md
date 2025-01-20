# Teach Me Anything: AI-Powered Teaching Assistant

Teach Me Anything is a Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, **ChromaDB**, and **Hugging Face**. This AI-powered tool helps users interact with subject-specific content and get concise, accurate, and relevant answers.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [File Structure](#file-structure)
5. [Setup Instructions](#setup-instructions)
6. [Code Explanation](#code-explanation)
7. [Concepts Explained](#concepts-explained)

---

## Overview

This application integrates **retrieval-based question answering (QA)** with **generative AI models** to provide insightful answers. Using LangChain and ChromaDB, it supports:

- Document ingestion for persistent storage.
- Retrieval of relevant content.
- Integration with Hugging Face language models for generation.
- Real-time interaction via Streamlit.

---

## Architecture

### Key Components:

1. **Student Model**:
   - Tracks user progress and stores historical interactions in relational databases.
   - Helps maintain a learner's profile.

2. **Expert System**:
   - Retrieves domain knowledge from vector stores (ChromaDB).
   - Utilizes LLMs (Language Models) for problem-solving.

3. **Tutoring Component**:
   - Content generation for queries.
   - Feedback mechanisms to refine responses.
   - Adaptive systems for personalized tutoring.

4. **User Interface (UI)**:
   - Built using Streamlit for real-time interaction.
   - Allows users to upload documents, ask questions, and explore topics.

---

## Features

1. **Document Ingestion**:
   - Upload `.txt` files for subject-specific knowledge.
   - Chunk and store documents in a vector database.

2. **Dynamic Model Selection**:
   - Choose from multiple Hugging Face models for text generation.

3. **Topic Management**:
   - Tracks explored topics and allows switching between them.

4. **Feedback Mechanism**:
   - Rate responses for continuous improvement.

5. **Chat History**:
   - Maintains session-based chat history for reference.

6. **Parameter Control**:
   - Adjust model parameters like `temperature`, `max_tokens`, and `top_p`.

---

## File Structure

```
Teaching_Assistant/
│
├── chroma_db/                   # Persistent storage for ChromaDB
├── files/                       # Directory for storing uploaded documents
├── Tutor.py                     # Module for managing RAG workflows
└── requirements.txt             # Python dependencies
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Teaching_Assistant
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run Tutor.py
```

---

## Code Explanation

Here’s a breakdown of the main `Tutor.py` file:

### **Initialization**

- **Imports**:
  - `faiss`, `numpy`, and `pickle` for vector database management.
  - `torch` for using Hugging Face models.
  - `LangChain` components for RAG pipelines.

- **Directories**:
  - `TEXTS_DIRECTORY` specifies the folder where uploaded files are stored.

- **Embedding Model**:
  - `HuggingFaceEmbeddings` initializes the embedding function.

```python
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

- **Vector Store**:
  - A ChromaDB instance is created for managing document embeddings.

```python
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)
```

---

### **Document Ingestion**

- **Chunking**:
  - Documents are split into manageable chunks to ensure efficient embedding.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

- **Loading**:
  - `.txt` files in the specified directory are loaded and added to the vector store.

```python
def load_and_chunk_documents_to_vector_store(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents = loader.load()
            chunked_documents = text_splitter.split_documents(documents)
            vector_store.add_documents(chunked_documents)
```

---

### **Streamlit UI**

#### **Sidebar**

- Topic Management:
  - Tracks all explored topics in `st.session_state.all_topics`.

- File Upload:
  - Allows users to upload `.txt` files for ingestion.

#### **Chat Interface**

- Accepts user queries via `st.text_input()`.
- Retrieves relevant documents using vector store similarity search.
- Constructs a prompt with context for the LLM.

```python
retrieved_documents = vector_store.similarity_search(user_input)
context = "\n".join([doc.page_content for doc in retrieved_documents])
```

- Generates a response using Hugging Face models.

---

### **Hugging Face Integration**

- **Model Loading**:
  - Dynamic selection of Hugging Face models via `st.selectbox()`.

```python
tokenizer, model = load_model(model_dict[selected_model_name])
```

- **Query Pipeline**:
  - Configured for text generation with user-defined parameters.

```python
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=float16,
    max_new_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    device="cpu"
)
```

- **RAG Pipeline**:
  - Combines retrieval (ChromaDB) and generation (Hugging Face).

```python
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=qa_model,
    retriever=vector_store.as_retriever()
)
```

---

### **Feedback and History**

- Users can rate responses and view/download their chat history.

---

## Contribution Guidelines

- Fork the repository.
- Create a new branch for your feature.
- Submit a pull request with a detailed description.

---

## License

This project is licensed under the MIT License.

---

## Concepts Explained

### What is an Embedding?

Embeddings are dense vector representations of text. They map text into a continuous vector space where similar texts are closer together. In this project, embeddings are generated using the `sentence-transformers/all-MiniLM-L6-v2` model to capture semantic relationships between documents and queries.

### What is a Vector Store?

A vector store is a database for storing embeddings along with metadata. It allows for efficient similarity searches. In this project, we use **ChromaDB** as the vector store for managing and retrieving document embeddings.

### What is ChromaDB?

**ChromaDB** is an open-source vector database. It helps store and query embeddings effectively, enabling the retrieval of semantically similar documents based on user queries.

### Why Use LangChain?

LangChain simplifies the development of retrieval-augmented generation (RAG) applications by providing tools for:
- Chunking and processing documents.
- Building pipelines for retrieval and generation.
- Integrating vector databases and language models seamlessly.

### Document Chunking

Chunking splits large documents into smaller parts to ensure embeddings capture fine-grained details. This approach enhances retrieval accuracy and relevance.

### What is Retrieval-Augmented Generation (RAG)?

RAG combines retrieval from a knowledge base (vector store) with generative language models. It ensures that answers are:
- Factually grounded.
- Relevant to user queries.
- Enhanced by contextual knowledge.

### Hugging Face Models

Hugging Face models are pre-trained language models used for:
- Text generation.
- Context-aware responses.
- Adaptability to various use cases.

### Feedback Mechanism

The feedback system allows users to rate responses, enabling continuous improvement of the system and tuning of model parameters.

---

