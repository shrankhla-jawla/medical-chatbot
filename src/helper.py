from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

#extract data from pdf file
def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf", 
        show_progress=True, 
        loader_cls=PyPDFLoader
        )
    documents = loader.load()
    return documents

#filter data
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    given a list of documnent objects, return a new list of document objects with 
    only 'SOURCE' in metadata and the original page content 
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src= doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return minimal_docs

#chunking operation
#split the documents into smaller chunks (for better processing)
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, #500 token one chunk
        chunk_overlap=20, #for understanding the context, we need to have some overlap between chunks
        )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

#download the embedding model and return it
def download_embeddings():
    """download and return the embedding model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

