from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import GigaChat
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

def vectorSearch(db, query):
    search_results = db.similarity_search(query, k=1)
    concut_result = '/n '.join([result.page_content for result in search_results])
    return concut_result
    
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
    if message.type != 'system':
        with st.chat_message(message.type):
            st.markdown(message.content)

promt = st.chat_input(placeholder='Введите сообщение')

if promt:
    st.session_state.messages = []
    relevant_search = vectorSearch(db=db, query=promt)
    st.session_state.messages.insert(0, SystemMessage(content=relevant_search))
    with st.chat_message('user'):
        st.markdown(promt)
    st.session_state.messages.append(HumanMessage(content=promt))
    response = giga(st.session_state.messages).content
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append(AIMessage(response))
     