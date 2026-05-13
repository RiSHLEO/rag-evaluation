import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def build_rag_pipeline(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, k: int = 3):
    """
    Build a RAG pipeline from a PDF with configurable parameters.
    Returns the chain and retriever separately so we can inspect retrieved chunks.
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    # Embed and store
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Prompt
    prompt = PromptTemplate(
        template="""You are a precise assistant. Your ONLY job is to answer the question using EXCLUSIVELY the information provided in the context below.

STRICT RULES:
1. Only use information explicitly stated in the context
2. If the exact answer is in the context, quote the relevant numbers or facts directly
3. Do not add any information from your general knowledge
4. If the context does not contain enough information, say exactly: "The context does not contain enough information to answer this question."
5. Keep your answer concise and directly focused on what was asked

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def get_answer_and_contexts(chain, retriever, question: str):
    """
    Run the RAG pipeline and return both the answer and the retrieved chunks.
    RAGAS needs both.
    """
    # Get retrieved chunks
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    # Get answer
    answer = chain.invoke(question)

    return answer, contexts