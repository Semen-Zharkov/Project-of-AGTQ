from langchain.embeddings import GPT4AllEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import Document
from langchain.vectorstores import FAISS


def filter_docs(docs: list[Document], question: str) -> list[Document]:
    embeddings = GPT4AllEmbeddings()
    retriever = FAISS.from_documents(docs, GPT4AllEmbeddings()).as_retriever()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

    filtered_docs = compression_retriever.get_relevant_documents(question)
    return filtered_docs
