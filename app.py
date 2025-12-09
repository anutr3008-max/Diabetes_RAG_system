import streamlit as st
import requests, zipfile, io, os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.title("🩺 Diabetes Healthcare RAG System (MedQuAD)")
st.write("Ask any diabetes-related medical question. The answer is generated using a RAG pipeline built from MedQuAD NIDDK data.")


# 1. Download & Load Dataset
@st.cache_data
def load_medquad():
    url = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
    response = requests.get(url)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("medquad_data")

    base_path = "medquad_data/MedQuAD-master"
    qa_pairs = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".xml"):
                file_path = os.path.join(root, file)
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for qa in root_xml.findall(".//QAPair"):
                    q = qa.findtext("Question", default="").strip()
                    a = qa.findtext("Answer", default="").strip()
                    f = qa.findtext("Focus", default="").strip()

                    if q and a:
                        qa_pairs.append({"question": q, "answer": a, "focus": f})

    df = pd.DataFrame(qa_pairs)

    # Filter for diabetes-related Q&A
    df_diabetes = df[
        df["question"].str.contains("diabetes", case=False, na=False) |
        df["answer"].str.contains("diabetes", case=False, na=False)
    ]

    return df_diabetes

df_diabetes = load_medquad()

st.success(f"Dataset Loaded: {df_diabetes.shape[0]} diabetes Q&A pairs")


# 2. Embeddings + FAISS

@st.cache_resource
def build_index(df):
    corpus = (df['question'] + " " + df['answer']).tolist()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = model.encode(corpus)

    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(emb))

    return model, index

model, index = build_index(df_diabetes)


# 3. RAG Generation Pipeline

generator = pipeline("text2text-generation",
                     model="google/flan-t5-base",
                     max_length=300)

def retrieve_docs(query, k=3):
    q_emb = model.encode([query])
    _, I = index.search(q_emb, k)
    return df_diabetes.iloc[I[0]]

def rag_answer(query):
    docs = retrieve_docs(query)

    context = ""
    for _, row in docs.iterrows():
        context += f"Q: {row['question']}\nA: {row['answer']}\n\n"

    prompt = f"""
    You are a medical assistant. Use ONLY the context below:

    Context:
    {context}

    Question: {query}

    Provide a medically accurate answer strictly from the context.
    """

    response = generator(prompt)[0]["generated_text"]

    return response, docs


# UI

st.subheader("Ask a Diabetes-related Question")

user_query = st.text_input("Enter your question here:")

if user_query:
    answer, docs = rag_answer(user_query)
    st.markdown("### 🩺 Answer:")
    st.write(answer)

    st.markdown("---")
    st.markdown("### 📚 Sources Used:")
    
    docs_display = docs[['question', 'answer']].copy()
   
    docs_display['answer'] = docs_display['answer'].apply(
    lambda x: x[:200] + ("..." if len(x) > 200 else "")
    )
    st.dataframe(docs_display)


