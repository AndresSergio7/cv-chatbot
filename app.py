import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# --- Setup API key ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Load documents ---
text_loader = TextLoader("about_me.txt")
pdf_loader = PyMuPDFLoader("Valleleal_Sergio_CV.pdf")
all_docs = text_loader.load() + pdf_loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(all_docs)

# --- Embeddings and Vectorstore ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# --- QA chain ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# --- Streamlit Page Settings ---
st.set_page_config(page_title="Sergio AI Chatbot", page_icon="🤖", layout="centered")

# --- Session State Setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Stylish Header ---
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

st.markdown("<h2 style='text-align:center;'>🤖 Sergio AI Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask me about my experience, projects or personal interests!</p>", unsafe_allow_html=True)

# --- Chat Display Area ---
st.markdown("<div class='message-container'>", unsafe_allow_html=True)
for q, a in st.session_state.chat_history:
    st.markdown(f"<div class='user-msg'>🧑‍💬 {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>🤖 {a}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Question Input ---
question = st.text_input("Type your message...", key="user_input")
if st.button("Send") and question:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(question)
        st.session_state.chat_history.append((question, answer))
        st.experimental_rerun()  # To immediately show new messages

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()


