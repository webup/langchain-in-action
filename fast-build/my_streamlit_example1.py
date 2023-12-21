from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

st.title('🦜🔗 中文小故事生成器')

prompt = ChatPromptTemplate.from_template("请编写一篇关于{topic}的中文小故事，不超过100字")
model = ChatOllama(model="llama2-chinese:13b")
chain = prompt | model

with st.form('my_form'):
    text = st.text_area('输入主题关键词:', '小白兔')
    submitted = st.form_submit_button('提交')
    if submitted:
        st.info(chain.invoke({"topic": text}))
