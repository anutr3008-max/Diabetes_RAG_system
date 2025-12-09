# 🩺 Diabetes Healthcare RAG System (MedQuAD)

Retrieval-Augmented Generation (RAG) system that answers diabetes-related medical questions using the MedQuAD dataset and a web interface deployed on Streamlit Cloud.

The system combines:

-Semantic search using sentence embeddings

-Fast document retrieval using FAISS

-Text generation using transformer-based models.


# 🌐 Live App

The application is deployed on Streamlit Community Cloud.

https://diabetes-rag-system.streamlit.app/


# 📁 Project Structure 

Diabetes-RAG-System/

    ├── app.py                  # Main Streamlit application

    ├── requirements.txt        # Python dependencies

    ├── README.md               # Project documentation


# ⚙️ Requirements

Libraries are required:

streamlit

sentence-transformers

transformers

faiss-cpu

pandas

requests

numpy

torch

These dependencies are listed in requirements.txt and are automatically installed by Streamlit Cloud during deployment.

# 🚀 How Deployment Works (Streamlit Cloud)
Deployed using Streamlit Community Cloud and GitHub:

1.The code is pushed to a GitHub repository.

2.The repository is connected to Streamlit Cloud.

3.Every time code is updated and pushed, the deployed app updates automatically.

4.No manual server setup is required.


#  🧠 How the System Works
1. Downloads and extracts MedQuAD dataset

2. Filters diabetes-related Q&A pairs

3. Converts text into semantic embeddings

4. Stores embeddings in a FAISS index

5. Retrieves relevant medical context using similarity search

6. Generates medically grounded answers using a transformer model
