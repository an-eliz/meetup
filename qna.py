from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import GigaChat
from langchain.chains import RetrievalQA
import streamlit as st


def splitDocuments():
    loader = TextLoader("./мастер_и_маргарита.md")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)
    return documents
    
embeddings = GigaChatEmbeddings(
    credentials=st.secrets.general.auth,
    verify_ssl_certs=False
)

giga = GigaChat(
    credentials=st.secrets.general.auth,
    model='GigaChat',
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS"
)

db = Chroma.from_documents(
    splitDocuments(),
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)


st.title('Чат бот')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

promt = st.chat_input(placeholder='Введите сообщение')

if promt:
    #st.session_state.messages = []
    with st.chat_message('user'):
        st.markdown(promt)
    st.session_state.messages.append(HumanMessage(content=promt))
    qa_chain = RetrievalQA.from_chain_type(giga, retriever=db.as_retriever())
    response = qa_chain.run({"query": promt})
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append(AIMessage(response))