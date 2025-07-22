import streamlit as st
import tiktoken
from loguru import logger
import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    st.set_page_config(page_title="MultiQuery RAG", page_icon="📚")
    st.title("_MultiQuery 기반 :red[문서 QA]_ 📚")

    with st.sidebar:
        uploaded_files = st.file_uploader("📎 문서를 업로드하세요", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
        process = st.button("📄 문서 처리")

        if st.button("🔍 API 키 테스트"):
            try:
                openai.api_key = openai_api_key
                resp = openai.Embedding.create(
                    model="text-embedding-3-small",
                    input=["테스트 문장"]
                )
                st.success("✅ API 키 정상입니다!")
            except Exception as e:
                st.error(f"❌ 실패: {e}")

    if process:
        if not openai_api_key:
            st.warning("API 키를 입력해주세요.")
            st.stop()

        docs = get_text(uploaded_files)
        chunks = get_text_chunks(docs)
        vectorstore = get_vectorstore(chunks, openai_api_key)
        chain = get_multiquery_chain(vectorstore, openai_api_key)
        st.session_state.conversation = chain
        st.session_state.chat_history = []

    if "conversation" in st.session_state:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(msg["question"])
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])

        if query := st.chat_input("질문을 입력하세요"):
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    response = st.session_state.conversation.invoke({"question": query})
                    st.markdown(response)

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": response
                })


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
        logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyMuPDFLoader(file_name)
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        doc_list.extend(loader.load_and_split())
    return doc_list


def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len
    )
    return [doc for doc in splitter.split_documents(docs) if doc.page_content.strip()]


def get_vectorstore(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(
        model_name="text-embedding-3-small",
        openai_api_key=api_key
    )
    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def get_multiquery_chain(vectorstore, api_key):
    llm = ChatOpenAI(
        model_name="gpt-4.1-2025-04-14",
        openai_api_key=api_key,
        temperature=0
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )

    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 책의 내용(context) 에서 주어진 질문(question) 에 답하는 것입니다.
    우선적으로 검색된 다음 책의 내용(context) 만을 사용하여 질문(question) 에 답하세요. 문서에 직접적인 설명이없더라도, 문맥을 통해 유추를 해보고 생각을 곁들여도 괜찮아. 그리고도 답을 모른다면 '잘 모르겠습니다' 라고 답하세요.
    한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

    그리고, 당신은 당신의 지능에 대한 신뢰성을 보여주기위해, 문맥에 의지하는것처럼 보이는건 피해야합니다. '문맥상...'으로 문장을 시작하면 당신이 당신의 대답에 대해 책임을 회피하려는것처럼 보이므로 이를 피해야합니다.

    #Context:
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    main()
