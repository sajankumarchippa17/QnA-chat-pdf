import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ("system", 
      "Answer the given question based on the context. "
      "Please provide an accurate answer from the context."
      "\n<context>\n{context}\n<context>"),
    ("human", "Question: {input}")
])

def create_vectors_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("Documents")
        st.session_state.docs=st.session_state.loader.load() 
        st.session_state.text_splitters=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.splitters=st.session_state.text_splitters.split_documents(st.session_state.docs)
        st.session_state.db=FAISS.from_documents(st.session_state.splitters, st.session_state.embeddings)
st.title("RAG QnA with documents")

user_prompt=st.text_input("Give the question you want to ask from all the pdfs")
if st.button("Document Embeddings Start"):
    create_vectors_embeddings()
    st.write("Embeddings are created ")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.db.as_retriever()
    retriever_chain=create_retrieval_chain(retriever, document_chain)
    response=retriever_chain.invoke({"input":user_prompt})
    st.write(response['answer'])

