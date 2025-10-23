import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma              # <-- antes: FAISS
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- splitter moderno
import os


# --- Streamlit Page Settings ---
st.set_page_config(page_title="Sergio AI Chatbot", page_icon="🤖", layout="centered")

# --- Session State Setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
# gpt-4o-mini cambiarlo a mini
# --- Setup API key and model ---
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# --- Load documents ---
text_loader = TextLoader("about_me.txt")
pdf_loader = PyMuPDFLoader("Valleleal_Sergio_CV_Español.pdf")
all_docs = text_loader.load() + pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(all_docs)

# --- Embeddings and Vectorstore ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=".chroma")
retriever = vectorstore.as_retriever()


# --- QA chain ---
prompt = ChatPromptTemplate.from_template(
    "Responde basado en el contexto:\n{context}\n\nPregunta: {input}"
)
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)



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
st.markdown("<p style='text-align:center;'>¡Pregúntame sobre mi experiencia, proyectos o intereses personales!</p>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 2em;'>Conoce más sobre mi experiencia y vida personal.</h1>
        <p style='font-size: 1.1em; max-width: 700px; margin: 0 auto;'>
            Hola, mi nombre es Sergio y creé este chatbot sencillo usando <b>LangChain</b>, <b>OpenAI</b> y <b>Streamlit</b>. Utiliza un modelo de lenguaje (LLM) para responder preguntas sobre mi experiencia profesional y personal basándose en mi currículum y en información personalizada.<br><br>
            <span style="color:gray;Por favor, recuerda que los tokens son limitados, no me agotes la tarjeta de crédito  🥲💸😂</span>
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
if st.button("Send") and st.session_state.get("user_input", ""):
    with st.spinner("Thinking..."):
        user_message = st.session_state["user_input"]
        result = qa_chain.invoke({"input": user_message})   # <-- invoke en vez de run
        answer = result.get("answer", "")
        st.session_state.chat_history.append((user_message, answer))

    del st.session_state["user_input"]
    st.rerun()



# --- Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
