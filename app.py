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
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(all_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

st.set_page_config(page_title="AI CV Chatbot", page_icon="🤖")
st.title("🤖 Ask Me About My Experience")

question = st.text_input("Ask me a question about my experience:")
if question:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(question)
        st.success("🤖 " + answer)
