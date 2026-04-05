from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import huggingfaceembeddings
from typing import List
from langchain.schema import Document

#extract data from pdf file
