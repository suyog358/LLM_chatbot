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
from langchain_core.embeddings import Embeddings


# Custom Embedding Wrapper that follows Langchain Embeddings interface
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()


# Groq API Key (consider moving to environment variable)
GROQ_API_KEY = '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'
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

st.title("Document Query System")

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)


# Vector Embedding Function
def vector_embedding():
    try:
        # Initialize Embeddings
        embeddings = SentenceTransformerEmbeddings()

        # Load PDF documents from a directory
        loader = PyPDFDirectoryLoader("C:/Users/suyog/OneDrive/Desktop/data_/data")  # Replace with your PDF directory
        docs = loader.load()

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])  # First 20 documents

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
        st.write("Vector Store DB Is Ready")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(f"Error details: {str(e)}")


# Streamlit UI
st.header("Document Query Interface")

# Document Embedding Button
if st.button("Create Vector Database"):
    vector_embedding()

# Query Input
prompt1 = st.text_input("Enter Your Question From Documents")

# Query Processing
def function(prompt1):
     if prompt1:
    # Check if vectors exist
        if "vectors" not in st.session_state:
            st.warning("Please create vector database first by clicking 'Create Vector Database'")
        else:
            try:
            # Create document and retrieval chains
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Process the query
                response = retrieval_chain.invoke({'input': prompt1})

            # Display Answer
                st.subheader("Answer:")
                st.write(response['answer'])

            # Show Similar Documents
                with st.expander("Document Similarity Search"):
                    for doc in response["context"]:
                        st.write(doc.page_content)
                        st.write("---")

            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.error(f"Error details: {str(e)}")
