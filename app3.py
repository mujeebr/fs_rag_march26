# import streamlit as st
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAI
# from langchain_classic.memory import ConversationBufferWindowMemory
# from langchain_classic.chains import ConversationalRetrievalChain
# from langchain_core.prompts import PromptTemplate

# st.title("Chat with PDFs")


# @st.cache_resource
# def load_and_process_pdfs():
#     """Load PDFs and create vectorstore once"""
#     loader = PyPDFDirectoryLoader("pdfs")
#     data = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_documents(data)
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(chunks, embedding_model)
#     return vectorstore


# QA_PROMPT = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""You are a friendly and conversational assistant. Use the document context below to answer questions when relevant. If the question is casual, a greeting, or unrelated to the documents, respond naturally and conversationally — do not say you cannot help.

# Context from documents:
# {context}

# Question: {question}

# Answer:""",
# )


# def build_chain(api_key):
#     """Build the ConversationalRetrievalChain with BufferWindowMemory (k=5)"""
#     vectorstore = load_and_process_pdfs()
#     llm = OpenAI(api_key=api_key)
#     memory = ConversationBufferWindowMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer",
#         k=5,
#     )
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT},
#         verbose=False,
#     )
#     return chain


# # Persist chain (and its memory) across Streamlit reruns
# if "chain" not in st.session_state:
#     st.session_state.chain = None
# if "chat_log" not in st.session_state:
#     st.session_state.chat_log = []

# st.sidebar.header("API Key")
# api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# # Re-initialize chain whenever the key changes
# if api_key and st.session_state.chain is None:
#     with st.spinner("Loading documents and building chain…"):
#         st.session_state.chain = build_chain(api_key)

# st.header("Ask a question about your PDFs")
# question = st.text_input("Enter your question here")
# generate_button = st.button("Generate Answer", key="generate_btn")

# if generate_button:
#     if not api_key:
#         st.warning("Please enter your OpenAI API key in the sidebar.")
#     elif not question:
#         st.warning("Please enter a question.")
#     else:
#         try:
#             with st.spinner("Generating answer…"):
#                 result = st.session_state.chain.invoke({"question": question})
#             answer = result.get("answer", result.get("result", ""))

#             st.session_state.chat_log.append({"question": question, "answer": answer})

#             st.subheader("Answer")
#             st.write(answer)

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# # Sidebar: show conversation log
# if st.session_state.chat_log:
#     st.sidebar.header("Conversation History (Last 5)")
#     for i, chat in enumerate(st.session_state.chat_log[-5:], 1):
#         st.sidebar.write(f"**Q{i}:** {chat['question']}")
#         st.sidebar.write(f"**A{i}:** {chat['answer']}")
#         st.sidebar.divider()

# import streamlit as st
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAI
# from langchain_classic.memory import ConversationBufferWindowMemory
# from langchain_classic.chains import ConversationalRetrievalChain
# from langchain_core.prompts import PromptTemplate

# st.title("Chat with PDFs (Conversational + Memory Enabled)")

# # -----------------------------
# # Load + Process PDFs
# # -----------------------------
# @st.cache_resource
# def load_and_process_pdfs():
#     loader = PyPDFDirectoryLoader("pdfs")
#     data = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     chunks = text_splitter.split_documents(data)

#     embedding_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     vectorstore = FAISS.from_documents(chunks, embedding_model)
#     return vectorstore


# # -----------------------------
# # Improved Prompt (IMPORTANT)
# # -----------------------------
# QA_PROMPT = PromptTemplate(
#     input_variables=["context", "question", "chat_history"],
#     template="""
# You are a friendly and conversational assistant.

# Rules:
# - If the question refers to previous conversation, use chat history.
# - If the question is about documents, use context.
# - If casual, respond naturally.
# - Never say you cannot help.

# Chat History:
# {chat_history}

# Document Context:
# {context}

# User Question:
# {question}

# Answer:
# """,
# )


# # -----------------------------
# # Build Chain (FIXED)
# # -----------------------------
# def build_chain(api_key):
#     vectorstore = load_and_process_pdfs()

#     llm = OpenAI(api_key=api_key, temperature=0.7)

#     memory = ConversationBufferWindowMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer",
#         k=5,
#     )

#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT},
#         get_chat_history=lambda h: h,   # 🔥 CRITICAL FIX
#         return_source_documents=False,
#         verbose=False,
#     )

#     return chain


# # -----------------------------
# # Session State
# # -----------------------------
# if "chain" not in st.session_state:
#     st.session_state.chain = None

# if "chat_log" not in st.session_state:
#     st.session_state.chat_log = []


# # -----------------------------
# # Sidebar
# # -----------------------------
# st.sidebar.header("API Key")
# api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# if api_key and st.session_state.chain is None:
#     with st.spinner("Loading PDFs and building chain..."):
#         st.session_state.chain = build_chain(api_key)


# # -----------------------------
# # Chat UI
# # -----------------------------
# st.header("Ask anything (Docs + Conversation supported)")
# question = st.text_input("Enter your question here")
# generate_button = st.button("Generate Answer")


# if generate_button:
#     if not api_key:
#         st.warning("Please enter API key")
#     elif not question:
#         st.warning("Please enter a question")
#     else:
#         try:
#             with st.spinner("Thinking..."):
#                 result = st.session_state.chain.invoke({"question": question})

#             answer = result.get("answer", "")

#             st.session_state.chat_log.append({
#                 "question": question,
#                 "answer": answer
#             })

#             st.subheader("Answer")
#             st.write(answer)

#         except Exception as e:
#             st.error(f"Error: {str(e)}")


# # -----------------------------
# # Chat History
# # -----------------------------
# if st.session_state.chat_log:
#     st.sidebar.header("Conversation History (Last 5)")
#     for i, chat in enumerate(st.session_state.chat_log[-5:], 1):
#         st.sidebar.write(f"**Q{i}:** {chat['question']}")
#         st.sidebar.write(f"**A{i}:** {chat['answer']}")
#         st.sidebar.divider()

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI

st.title("Chat with PDFs (True Conversational Memory)")

# -----------------------------
# Load PDFs
# -----------------------------
@st.cache_resource
def load_vectorstore():
    loader = PyPDFDirectoryLoader("pdfs")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Build LLM
# -----------------------------
def get_llm(api_key):
    return OpenAI(api_key=api_key, temperature=0.7)


# -----------------------------
# Manual RAG + Memory
# -----------------------------
def generate_answer(llm, retriever, question, chat_history):

    # 🔥 Convert history to strong text format
    history_text = ""
    for chat in chat_history[-5:]:
        history_text += f"User: {chat['question']}\nAssistant: {chat['answer']}\n"

    # 🔍 Retrieve docs
    # docs = retriever.get_relevant_documents(question)
    # context = "\n\n".join([doc.page_content for doc in docs[:5]])
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    # 🧠 Strong prompt
#     prompt = f"""
# You are a highly intelligent conversational assistant.

# IMPORTANT RULES:
# - You MUST prioritize chat history when the question refers to past conversation.
# - If user says "my", "their", "who", ALWAYS check chat history first.
# - NEVER say "as an AI" or deny memory.
# - If info exists in chat history, use it confidently.
# - Use document context only if relevant.

# CHAT HISTORY:
# {history_text}

# DOCUMENT CONTEXT:
# {context}

# USER QUESTION:
# {question}

# ANSWER:
# """

    prompt = f"""
    You are a natural, friendly conversational assistant.

    IMPORTANT RULES:
    - Speak like a human, not like a system.
    - NEVER say phrases like:
    "based on chat history", "according to context", "as an AI"
    - Just answer naturally as if you remember the conversation.
    - ensure you use chat history to answer the questions, if you don't find an answer there move to the context
    - If the user shares personal info, remember and use it naturally.
    - Keep responses conversational and concise.
    - never say "I don't know" or "I'm not sure"

    CHAT HISTORY:
    {history_text}

    DOCUMENT CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Answer naturally:
    """


    response = llm.invoke(prompt)
    return response


# -----------------------------
# Session State
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key and st.session_state.vectorstore is None:
    with st.spinner("Processing PDFs..."):
        st.session_state.vectorstore = load_vectorstore()


# -----------------------------
# Chat UI
# -----------------------------
st.header("Ask anything")
question = st.text_input("Enter your question")
btn = st.button("Generate Answer")

if btn:
    if not api_key:
        st.warning("Enter API key")
    elif not question:
        st.warning("Enter a question")
    else:
        try:
            llm = get_llm(api_key)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

            answer = generate_answer(
                llm,
                retriever,
                question,
                st.session_state.chat_log
            )

            st.session_state.chat_log.append({
                "question": question,
                "answer": answer
            })

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# History
# -----------------------------
if st.session_state.chat_log:
    st.sidebar.header("Conversation History")

    for i, chat in enumerate(st.session_state.chat_log[-5:], 1):
        st.sidebar.write(f"**Q{i}:** {chat['question']}")
        st.sidebar.write(f"**A{i}:** {chat['answer']}")
        st.sidebar.divider()