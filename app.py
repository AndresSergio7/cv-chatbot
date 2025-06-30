import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

text_loader = TextLoader("about_me.txt")
text_docs = text_loader.load()

pdf_loader = PyMuPDFLoader("Valleleal_Sergio_CV.pdf")
pdf_docs = pdf_loader.load()

all_docs = pdf_docs + text_docs
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(all_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("🚀 Trying to embed documents...")
try:
    vectorstore = FAISS.from_documents(docs, embeddings)
except Exception as e:
    st.error(f"Embedding error: {e}")
    st.stop()

vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

st.set_page_config(page_title="SERGIO AI Chatbot", page_icon="🤖")
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 3em;'>🤖 Ask Me About My Experience or Personal Life</h1>
        <p style='font-size: 1.2em; max-width: 700px; margin: 0 auto;'>
            This chatbot was built using <b>LangChain</b>, <b>OpenAI</b>, and <b>Streamlit</b>. It uses a language model (LLM) to answer questions about my professional and personal experience based on my resume and custom input.<br><br>
            <span style="color:gray;">Please remember that tokens are limited — don’t max out my credit card 🥲💸😂</span>
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("### 💡 Try asking one of these:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📈 Tell me about an AI project"):
        st.session_state["preset_question"] = "Can you tell me about a project you've done in AI?"

with col2:
    if st.button("🏢 What companies have you worked for?"):
        st.session_state["preset_question"] = "What companies have you worked for?"

with col3:
    if st.button("🌍 What languages do you speak?"):
        st.session_state["preset_question"] = "What languages do you speak?"


question = st.text_input(
    "Ask me something:",
    value=st.session_state.get("preset_question", "")
)

# Clear preset after use
st.session_state["preset_question"] = ""

