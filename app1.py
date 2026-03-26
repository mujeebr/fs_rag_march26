import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

st.title("Chat with pdfs")

@st.cache_resource
def load_and_process_pdfs():
    """Load PDFs and create vectorstore once"""
    loader = PyPDFDirectoryLoader("pdfs")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Enter your OPENAI API key", type="password")

st.header("Ask a question about your pdfs")
question = st.text_input("Enter your question here")

if api_key and question:
    try:
        vectorstore = load_and_process_pdfs()
        llm = OpenAI(api_key=api_key)
        
        template = """Use the context to provide a concise answer and if you don't know just say you don't know.
        {context}
        Question: {question}
        Answer: """
        
        prompt = PromptTemplate.from_template(template)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        with st.spinner("Generating answer..."):
            response = chain.run(question)
        st.subheader("Here is your Answer")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")