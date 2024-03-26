from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import GigaChat
from langchain.chat_models.gigachat import GigaChat
from langchain.chat_models import GigaChat
import streamlit as st

giga = GigaChat(
    credentials=st.secrets.general.auth,
    model='GigaChat',
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS"
)


st.title('Чат бот')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

promt = st.chat_input(placeholder='Введите сообщение')
if promt:
    with st.chat_message('user'):
        st.markdown(promt)
    st.session_state.messages.append(HumanMessage(content=promt))
    response = giga(st.session_state.messages).content
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append(AIMessage(response))