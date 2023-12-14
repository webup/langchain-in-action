from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

from fastapi import FastAPI
from langserve import add_routes


# 设定系统上下文，构建提示词
template = """请扮演一位资深的技术博主，您将负责为用户生成适合在微博发送的中文帖文。
请把用户输入的内容扩展成 140 字左右的文字，并加上适当的 emoji 使内容引人入胜并专业。"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

# 通过 Ollama 加载 Llama 2 中文增强模型
model = ChatOllama(model="llama2-chinese")

# 通过 LCEL 构建调用链
chain = prompt | model | StrOutputParser()

# 构建 FastAPI 应用
app = FastAPI(
    title="微博技术博主",
    description="基于 LangChain 构建并由 LangServe 部署的微博技术博主 API"
)

# 通过 LangServe 将 chain 加入 writer 这一 API 路径
add_routes(app, chain, path="/writer")


# 主程序运行 Unicorn 服务端
if __name__ == "__main__":
    import uvicorn
    # 通过调整 host="0.0.0.0" 可将本地 API 服务暴露给其他设备访问
    uvicorn.run(app, host="localhost", port=8000)