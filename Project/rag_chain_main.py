from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xWa****tZZ****LFu****UzM" #my acess token
def load_transcript_and_index(video_id: str):
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    rag_chain = build_rag_chain(retriever)

    return retriever, rag_chain


def build_rag_chain(retriever):
    
    llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
    model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        ONLY answer using the context below.
        If the context is insufficient, say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser


    return main_chain
