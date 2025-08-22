import streamlit as st
import tiktoken
import openai
import time
import math

from loguru import logger
from openai.error import RateLimitError

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
        uploaded_files = st.file_uploader(
            "📎 문서를 업로드하세요", type=["pdf", "docx", "pptx"], accept_multiple_files=True
        )
        openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")

        process = st.button("📄 문서 처리")
    
        st.markdown("### ⚙️ 임베딩 / 청크 설정")
        max_chunks = st.number_input(
            "🔢 임베딩할 최대 청크 수 (개)",
            min_value=50,
            max_value=20000,
            value=800,
            step=50,
            help="업로드한 문서에서 생성된 청크 중 상위 N개만 임베딩합니다.",
        )
        chunk_size_ui = st.number_input(
            "🧩 청크 크기 (토큰 근사)",
            min_value=200,
            max_value=3000,
            value=800,
            step=50,
            help="청크가 작을수록 임베딩 당 토큰 수가 줄어듭니다.",
        )
        chunk_overlap_ui = st.number_input(
            "🔁 청크 오버랩",
            min_value=0,
            max_value=1000,
            value=120,
            step=10,
            help="인접 청크 간 중복 토큰 수.",
        )

        st.markdown("### 🐢 레이트리밋 내성")
        batch_size = st.number_input(
            "📦 임베딩 배치 크기",
            min_value=8,
            max_value=512,
            value=64,
            step=8,
            help="이 배치 단위로 OpenAI Embeddings API를 호출합니다.",
        )
        pause_seconds = st.number_input(
            "⏱️ 배치 간 대기 (초)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help="배치 간 잠깐 쉬어 OpenAI 레이트리밋을 피합니다.",
        )


        if st.button("🔍 API 키 테스트"):
            try:
                # v0.28.x 스타일 (현재 코드와 동일 계열)
                openai.api_key = openai_api_key
                _ = openai.Embedding.create(
                    model="text-embedding-3-large", input=["테스트 문장"]
                )
                st.success("✅ API 키 정상입니다!")
            except Exception as e:
                st.error(f"❌ 실패: {e}")

    if process:
        if not openai_api_key:
            st.warning("API 키를 입력해주세요.")
            st.stop()

        if not uploaded_files:
            st.warning("하나 이상의 문서를 업로드하세요.")
            st.stop()

        # 1) Load & split documents
        docs = get_text(uploaded_files)
        if not docs:
            st.error("문서를 읽지 못했습니다. 지원 형식을 확인해주세요.")
            st.stop()

        chunks = get_text_chunks(
            docs, chunk_size=chunk_size_ui, chunk_overlap=chunk_overlap_ui
        )
        if not chunks:
            st.error("추출된 텍스트가 없습니다.")
            st.stop()

        if len(chunks) > max_chunks:
            chunks = chunks[: int(max_chunks)]
            st.info(f"총 청크가 많아 상위 {len(chunks)}개만 임베딩합니다.")

        # 2) Build vector store with throttled batching
        with st.spinner("임베딩 및 인덱스 생성 중..."):
            vectorstore = get_vectorstore(chunks, openai_api_key)
        st.success("✅ 인덱스 생성 완료!")

        # 3) Build retriever chain
        chain = get_multiquery_chain(vectorstore, openai_api_key)
        st.session_state.conversation = chain
        st.session_state.chat_history = []

    # Chat UI
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
                    try:
                        response = st.session_state.conversation.invoke(
                            {"question": query}
                        )
                    except Exception as e:
                        response = f"오류가 발생했습니다: {e}"
                    st.markdown(response)

                st.session_state.chat_history.append(
                    {"question": query, "answer": response}
                )




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


def get_text_chunks(docs, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
    )
    return [doc for doc in splitter.split_documents(docs) if doc.page_content.strip()]


def get_vectorstore(text_chunks, api_key):
    """
    Build FAISS index in small batches with live ETA.
    - Shows a Streamlit progress bar with remaining time.
    - Handles OpenAI rate limits via backoff.
    """
    # You can tune these without changing any other code:
    BATCH_SIZE = 64
    PAUSE_BETWEEN_BATCHES = 1.5  # seconds

    embeddings = OpenAIEmbeddings(
        model_name="text-embedding-3-small",
        openai_api_key=api_key
    )

    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]
    total = len(texts)
    if total == 0:
        return FAISS.from_texts(texts=[], embedding=embeddings, metadatas=[])

    # Helpers
    def _fmt_eta(sec: float) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    progress = st.progress(0.0, text=f"임베딩/인덱싱 시작... (0/{total})")
    start_t = time.perf_counter()

    vs = None
    i = 0
    ema_per_item = None        # exponential moving average of seconds per item
    alpha = 0.25               # smoothing factor

    while i < total:
        j = min(i + BATCH_SIZE, total)
        batch_texts = texts[i:j]
        batch_metas = metadatas[i:j]

        batch_start = time.perf_counter()
        backoff = 2.0

        while True:
            try:
                if vs is None:
                    vs = FAISS.from_texts(
                        texts=batch_texts,
                        embedding=embeddings,
                        metadatas=batch_metas
                    )
                else:
                    vs.add_texts(texts=batch_texts, metadatas=batch_metas)
                break
            except RateLimitError:
                time.sleep(backoff)
                backoff = min(30.0, backoff * 1.8)

        # Optional gentle pacing between batches
        if j < total:
            time.sleep(PAUSE_BETWEEN_BATCHES)

        # Update timing stats
        batch_time = time.perf_counter() - batch_start
        items = j - i
        per_item = batch_time / max(1, items)
        ema_per_item = per_item if ema_per_item is None else (
            alpha * per_item + (1 - alpha) * ema_per_item
        )

        done = j
        remaining = total - done
        # Add expected sleep cost for remaining batches
        remaining_batches = math.ceil(remaining / BATCH_SIZE) if BATCH_SIZE else 0
        eta_sec = (ema_per_item * remaining) + (PAUSE_BETWEEN_BATCHES * remaining_batches)

        frac = done / total
        progress.progress(
            frac,
            text=f"임베딩/인덱싱 진행 중... {done}/{total} · ETA {_fmt_eta(eta_sec)}"
        )

        i = j

    total_time = time.perf_counter() - start_t
    progress.progress(1.0, text=f"임베딩/인덱싱 완료! 총 소요 {_fmt_eta(total_time)}")
    return vs

    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def get_multiquery_chain(vectorstore, api_key):
    llm = ChatOpenAI(
        model_name="gpt-5",
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
