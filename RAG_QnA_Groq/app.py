import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
# faiss does semantics search internally for the input arguments

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# we use the above for embedding or converting the test to vectors 
from dotenv import load_dotenv

load_dotenv()

# load the groq and google gem dependencies

groq_api_key = os.environ.get('GROOQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

st.title('RAG Based Document Q/A')

llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name = "gemma-7b-it",
        verbose=True,
)

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


def vector_embedding():
    """
    we will read all the docs from the pdf
    and convert them into vectors using the Google Generative AI Embeddings
    after converting them into chunks 
    and keep them into vector store FAISS
    and keep them in the session state so that we can access 

    """               
    # session state will allow us to use it anywhere we want

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        ## Data Ingestion
        st.session_state.loader=PyPDFDirectoryLoader("./pdf_data")
         ## Document Loading
        st.session_state.docs=st.session_state.loader.load()  
        ## Chunk Creation
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        #splitting
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) 
        #vecto embeddings
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) 


prompt1 = st.text_input('Please enter your querry.......')

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB created successfully")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    strart= time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    # print("Response time :",time.process_time()-start)
    st.write(response['answer'])
   
    # gemma also gives the context also in return so we can display that also using below code
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")