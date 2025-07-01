import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# --- Streamlit Page Settings ---
st.set_page_config(page_title="Sergio AI Chatbot", page_icon="🤖", layout="centered")

# --- Session State Setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# --- Setup API key and model ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# --- Load documents ---
text_loader = TextLoader("about_me.txt")
pdf_loader = PyMuPDFLoader("Valleleal_Sergio_CV.pdf")
all_docs = text_loader.load() + pdf_loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(all_docs)

# --- Embeddings and Vectorstore ---
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(docs, embeddings)

# --- QA chain ---
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# --- Custom Styles ---
st.markdown("""
    <style>
        .message-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 1rem;
            background-color: #f4f4f4;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .user-msg {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
        }
        .bot-msg {
            background-color: #FFF;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: left;
            border-left: 4px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h2 style='text-align:center;'>🤖 Sergio AI Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask me about my experience, projects or personal interests!</p>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 2em;'>Get to know more about my experience and personal life</h1>
        <p style='font-size: 1.1em; max-width: 700px; margin: 0 auto;'>
            Hi, my name is Sergio and I created this simple chatbot using <b>LangChain</b>, <b>OpenAI</b>, and <b>Streamlit</b>. It uses a language model (LLM) to answer questions about my professional and personal experience based on my resume and custom input.<br><br>
            <span style="color:gray;">Please remember that tokens are limited — don’t max out my credit card 🥲💸😂</span>
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Display Chat History ---
st.markdown("<div class='message-container'>", unsafe_allow_html=True)
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"<div class='user-msg'>🧑‍💬 {user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>🤖 {bot_msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Question Input ---
question = st.text_input("Type your message...", key="user_input")

# --- Handle Send ---
if st.button("Send") and st.session_state.user_input:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(st.session_state.user_input)
        st.session_state.chat_history.append((st.session_state.user_input, answer))
        if "user_input" in st.session_state and st.session_state["user_input"]:
            st.session_state["user_input"] = ""  # ✅ Safe reset
    st.rerun()


# --- Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
