import streamlit as st
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

# Load the GROQ API KEY
groq_api_key = os.getenv('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')

st.title("Chatgroq With Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# Use SentenceTransformers for embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        # Use SentenceTransformer for embedding instead of OpenAI
        st.session_state.embeddings = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with desired model
        st.session_state.loader = PyPDFDirectoryLoader("C:/Users/suyog/OneDrive/Desktop/data_/data")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20])  # Splitting

        # Create embeddings for documents
        docs_embeddings = [
            st.session_state.embeddings.encode(doc.page_content)
            for doc in st.session_state.final_documents
        ]

        # Initialize FAISS vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            docs_embeddings
        )


prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
