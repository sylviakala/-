import io
import streamlit as st
# from PIL import Image

from agent import ConversationAgent

st.title('🤖AI移动日记😜')


# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

if "welcome_word" not in st.session_state:
    st.session_state.welcome_word = "今天过的怎么样呢？"
    st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.welcome_word['text']})
    st.session_state.agent = ConversationAgent()
    st.session_state.agent.seed_agent()
    st.session_state.agent.generate_stage_analyzer(verbose=True)

# 展示聊天记录
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar='☺️'):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar='🤖'):
            st.markdown(message["content"])

# 用于用户输入
if prompt := st.chat_input('我们来聊聊今天发生了什么吧'):

    with st.chat_message('user', avatar='☺️'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    st.session_state.agent.determine_conversation_stage(prompt)
    st.session_state.agent.human_step(prompt)
    response = st.session_state.agent.step()

    with st.chat_message('assistant', avatar='🤖'):
        st.markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
