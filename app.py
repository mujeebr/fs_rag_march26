# import all libraries

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import streamlit as st



# set the app title
st.title("Chat with pdfs")
# create a side bar for api key
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Enter your OPENAI API key", type="password")


# Input section
st.header("Ask a question about your pdfs")
question = st.text_input("Enter your question here")

# check if the api_key

if api_key:
    # try:
        # extract the info from the pdfs
        loader = PyPDFDirectoryLoader("pdfs")
        data = loader.load()

        # create the chunks from the parsed info
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)

        # define the embedding model from huggingface
        embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

        # dump the embeddings into a vectorstore
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        # define the llm
        llm = OpenAI(api_key=api_key)

        # set up the prompt template
        template = """ Use the context to provide a concise answer and if you don't know just say don't now.
        {context}
        Question: {question}
        Answer: """

        Prompt = PromptTemplate.from_template(template)

        # set up the retrieval qa chain
        chain = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff",
                                    retriever = vectorstore.as_retriever(search_kwargs={"k": 15}),chain_type_kwargs = {"prompt":Prompt})
        
        if question:
            with st.spinner("Generating answer..."):
                response = chain.run(question)
            st.subheader("Here is your Answer")
            st.write(response)


    # except Exception as e:
    #     st.error("Error while processing pdfs and generating a response")