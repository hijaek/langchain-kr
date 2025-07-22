import streamlit as st
import tiktoken
from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def main():
    st.set_page_config(page_title="DirChat", page_icon="📚")
    st.title("_Private Data :red[QA Chat]_ 📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("📄 문서 처리")

        # ✅ 최신 OpenAI 방식으로 임베딩 API 테스트
        if st.button("🔍 API 키 테스트"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=["테스트 문장입니다"]
                )
                st.success("✅ 임베딩 호출 성공! 키 유효함")
            except Exception as e:
                st.error(f"❌ API 호출 실패: {e}")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks, openai_api_key)
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for i, doc in enumerate(source_documents[:3]):
                        st.markdown(doc.metadata.get("source", f"Document {i+1}"), help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})


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
            documents = loader.load()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        else:
            documents = []

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len
    )
    chunks = splitter.split_documents(text)
    return [doc for doc in chunks if doc.page_content and doc.page_content.strip()]


def get_vectorstore(text_chunks, openai_api_key):
    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]

    embeddings = OpenAIEmbeddings(
        model_name="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4.1-2025-04-14",
        temperature=0
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )


if __name__ == '__main__':
    main()
