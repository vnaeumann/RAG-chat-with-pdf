import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os


def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100)
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    
def handle_userinput(question):
    if st.session_state.convo is None:
        st.warning("Please upload and process PDFs first.")
        return

    response = st.session_state.convo({'question': question})
    st.session_state.chat_history = response['chat_history']
    
def display_chat():
    for message in st.session_state.chat_history:
        if message.type == "human":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <span class="chat-label">🧑 You</span>
                    <div>{message.content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif message.type == "ai":
            st.markdown(
                f"""
                <div class="chat-message bot">
                    <span class="chat-label">🤖 Bot</span>
                    <div>{message.content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def clear_chat():
    st.session_state.chat_history = []

    if st.session_state.convo is not None:
        st.session_state.convo.memory.clear()
        
        


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon=":books:"
    )
    load_css()
    
    if "convo" not in st.session_state:
        st.session_state.convo = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""


    st.header("Chat with Multiple PDFs :books:")


    user_question = st.text_input("Ask a question", key="user_question")
    if user_question and st.session_state.convo:
        handle_userinput(user_question)
    display_chat()


    with st.sidebar:
        st.subheader("Your documents")
        

        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.success("Text extracted")
                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.success("Text chunked")
                
                # create vector store
                vector_store = get_vectorstore(text_chunks)
                st.success("Vector store created")
                
                st.session_state.convo = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()