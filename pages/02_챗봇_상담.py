import streamlit as st
import json
import os
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage

batch_size = 50


def load_jsonl(file_path: str) -> str:
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            datas = json.loads(line)
            for data in datas:
                role, message = data.values()
                documents.append(f"{role}: {message}")
    return "\n".join(documents[:batch_size])


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "ai", "message": "안녕하세요. 어떤 고민으로 오셨나요?"}
    ]


st.set_page_config(
    page_title="챗봇 상담",
    page_icon="🦜",
)


path = "data/total_kor_multiturn_counsel_bot.jsonl"
text = load_jsonl(path)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n", ".", "?"],
)
chunks = splitter.split_text(text)
docs = [Document(page_content=chunk) for chunk in chunks]
embedding = OpenAIEmbeddings()
cache_path = ".cache/chatbot"
os.makedirs(cache_path, exist_ok=True)
cache_dir = LocalFileStore(cache_path)
cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedding, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo",
)


@st.cache_resource
def load_memory():
    return ConversationSummaryMemory(llm=llm)


memory = load_memory()


def load_history(_) -> list[BaseMessage]:
    return memory.chat_memory.messages


def save_history(role: str, message: str):
    st.session_state["messages"].append({"role": role, "message": message})


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 사람의 감정을 깊이 이해하고, 따뜻하게 질문을 던지는 심리 상담 챗봇입니다.
내담자의 감정을 평가하지 말고, 그 감정을 더 잘 이해할 수 있도록 부드럽게 질문을 이어가세요.
답변은 **반드시 2문장 이하로 짧게** 하고, 마지막은 항상 열린 질문으로 마무리하세요.
설명이나 조언보다는 내담자가 스스로 감정을 말할 수 있도록 이끌어 주세요.
다음은 지금까지의 대화입니다:

{context}
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
chain = (
    {
        "context": retriever,
        "history": RunnableLambda(load_history),
        "question": RunnablePassthrough(),
    }
    | template
    | llm
)


def invoke_chain(question: str) -> None:
    display_message("human", question)
    result = chain.invoke(question)
    display_message("ai", result.content)
    save_history("human", question)
    save_history("ai", result.content)


def display_message(role, message):
    with st.chat_message(role):
        st.write(message)


for chat in st.session_state["messages"]:
    display_message(chat["role"], chat["message"])

question = st.chat_input("고민이 있으신가요? 이곳에 물어보세요!")

if question:
    invoke_chain(question)
