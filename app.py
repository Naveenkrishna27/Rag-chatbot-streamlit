# app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Caching the chain so it doesn't re-load every time
@st.cache_resource
def load_chain():
    loader = CSVLoader(file_path='faqs.csv')
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings)

    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

# Load the RAG pipeline
qa_chain = load_chain()

# Streamlit UI
st.title("🤖 RAG Chatbot (LangChain + HuggingFace)")
st.markdown("Ask a question based on the knowledge base (FAQ in CSV).")

query = st.text_input("You:", placeholder="E.g., What is RAG?")
if st.button("Ask"):
    if query:
        response = qa_chain.run(query)
        st.success(f"Bot: {response}")

        # Save the Q&A to a file
        with open("qa_samples.txt", "a") as f:
            f.write(f"Q: {query}\nA: {response}\n\n")
    else:
        st.warning("Please enter a question to ask.")

