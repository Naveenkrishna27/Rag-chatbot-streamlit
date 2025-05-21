import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Cache chatbot setup to avoid reloading
@st.cache_resource
def load_chain():
    loader = CSVLoader(file_path='faqs.csv')
    documents = loader.load()

    # HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # HuggingFace LLM
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

qa_chain = load_chain()

# Streamlit UI
st.title("🤖 RAG Chatbot (LangChain + HuggingFace + FAISS)")
st.write("Ask a question based on the custom FAQ dataset.")

query = st.text_input("You:", placeholder="e.g., What is RAG?")
if st.button("Ask"):
    if query:
        answer = qa_chain.run(query)
        st.success(f"Bot: {answer}")

        # Optional: Save to log file
        with open("qa_samples.txt", "a") as f:
            f.write(f"Q: {query}\nA: {answer}\n\n")
    else:
        st.warning("Please enter a question to get an answer.")
