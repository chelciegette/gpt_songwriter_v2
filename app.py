import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="🎶 GPT Songwriter", layout="centered")
st.title("🎶 GPT Songwriter Assistant")
st.write("Ask questions based on your notes, lyrics, or chord ideas!")

# Load documents from songwriter_data folder
docs = []
data_folder = Path("songwriter_data")
if data_folder.exists():
    for file in data_folder.glob("*.txt"):
        loader = TextLoader(str(file))
        docs.extend(loader.load())
else:
    st.warning("No 'songwriter_data' folder found!")

# Only continue if documents exist
if docs:
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # Set up GPT and memory
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    # User input
    user_input = st.text_input("What would you like help with?")

    if user_input:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_input)
        st.markdown(f"**Response:** {response}")
else:
    st.info("Add some .txt files to the 'songwriter_data' folder to get started.")
