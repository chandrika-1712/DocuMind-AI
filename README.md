# DocuMind AI — Intelligent Document Analysis & RAG System

DocuMind AI is an AI-powered application that helps users understand and explore documents using Natural Language Processing (NLP), Machine Learning, semantic search, and Retrieval-Augmented Generation (RAG).

Upload a PDF or TXT document to extract its text, generate an AI-powered summary, classify its content, retrieve relevant passages, and ask questions based on the document.

## ✨ Features

- **Document Upload:** Supports PDF and TXT files.
- **Text Extraction:** Extracts document content and displays basic text statistics.
- **AI Summarization:** Generates concise summaries using Google Gemini.
- **Document Classification:** Predicts whether a document belongs to Technology, Science, or Sports.
- **Semantic Search:** Finds relevant passages using Sentence Transformers and FAISS.
- **RAG-Based Question Answering:** Answers questions using retrieved document passages and provides source references.
- **Passage Exploration:** Displays relevant passages and similarity scores.
- **Download Extracted Text:** Allows users to download extracted document content.

## 🛠️ Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **Machine Learning:** Scikit-learn, Logistic Regression, Linear SVM
- **NLP & Embeddings:** Sentence Transformers
- **Vector Search:** FAISS
- **Generative AI:** Google Gemini API
- **Document Processing:** PDF and TXT text extraction

## 🧠 Machine Learning Model

The document classifier is trained using a selected subset of the Scikit-learn 20 Newsgroups dataset, grouped into three categories:

- Technology
- Science
- Sports

Two classification algorithms were evaluated:

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 93.62% | 92.84% |
| Linear SVM | 94.33% | 93.73% |

**Best-performing model:** Linear SVM.

The reported metrics are from the project's held-out test set. They describe performance on this selected dataset and do not guarantee the same accuracy on arbitrary uploaded documents.

## 🔎 How RAG Works

1. The user uploads a document.
2. The application extracts its text and divides it into overlapping passages.
3. Sentence Transformers converts the passages into numerical embeddings.
4. FAISS retrieves passages that are semantically similar to the user's question.
5. The retrieved passages are provided to Google Gemini as context.
6. Gemini generates an answer grounded in the supplied passages, with source references.

This approach helps users find information in documents without relying only on the language model's general knowledge.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- A Google Gemini API key for AI summarization and question answering

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DocuMind-AI
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file, run:

```powershell
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key.

**Security:** Never commit your `.env` file or publish your API key.

### 5. Run the application

```powershell
streamlit run app.py
```

The application will open in your browser, where you can upload documents and explore its features.

## 📁 Project Structure

```text
DocuMind-AI/
├── app.py
├── gemini_engine.py
├── rag_engine.py
├── train_classifier.py
├── test_rag.py
├── artifacts/
│   ├── document_classifier.joblib
│   └── confusion_matrix.png
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Model Evaluation

The project evaluates classification models using accuracy, macro F1-score, and a confusion matrix. The saved best-performing pipeline is loaded by the application for document classification.

The RAG pipeline uses semantic similarity to retrieve relevant passages before generating answers. Its output depends on document content, passage retrieval quality, and the configured language model.

## 🎯 Use Cases

- Understanding lengthy documents
- Quickly summarizing reports and notes
- Searching documents using natural-language questions
- Exploring document content through relevant passages
- Automatically categorizing text documents

## 🚀 Future Enhancements

- Support for multiple documents and document collections
- Improved retrieval using hybrid search and reranking
- More comprehensive retrieval and answer-quality evaluation
- Additional document categories and datasets
- User authentication and document management

## 👩‍💻 Project Objective

DocuMind AI demonstrates the integration of machine learning, NLP, vector-based semantic retrieval, and generative AI in a practical document intelligence application.

---

**Technologies:** Python · Streamlit · Scikit-learn · NLP · Sentence Transformers · FAISS · Google Gemini · RAG
