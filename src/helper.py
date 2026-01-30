from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Extracting text from PDF files in a directory
def load_medical_pdfs(folder_path):
    """
    Loads all medical PDF files from the specified folder path and returns a list of documents.
    
    """
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

#Splitting text into smaller chunks
from typing import List
from langchain.schema import Document
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Given a list of documents objects return a list of documents with only the page_content attribute populated and source in metadata."""
    
    minimal_docs:List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source","")
        minimal_docs.append(
             Document(
            page_content=doc.page_content,
            metadata={"source":src}
        )
        )
    return minimal_docs


#Text Splitting Example

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk



from langchain_community.embeddings import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    """
    Downloads and loads the embedding model used for medical documents
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
