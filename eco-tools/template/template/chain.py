from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser


# 设定系统上下文，构建提示词
template = """请扮演一位资深的技术博主，您将负责为用户生成适合在微博发送的中文帖文。
请把用户输入的内容扩展成 140 字左右的文字，并加上适当的 emoji 使内容引人入胜并专业。"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

# 通过 Ollama 加载 Llama 2 中文增强模型
model = ChatOllama(model="llama2-chinese")

# 通过 LCEL 构建调用链
chain = prompt | model | StrOutputParser()