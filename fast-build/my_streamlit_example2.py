from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

openai_api_key = st.sidebar.text_input('OpenAI API Key')

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, streaming=True)
    tools = load_tools(["ddg-search"])
    # 创建 Agent
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # 通过回调方式展示 LLM 思考和行为
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
