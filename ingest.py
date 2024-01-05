from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    print("Starting the function")
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    print("Loading the document")
    documents = loader.load()
    print(documents)
    print("Splitting the text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    print("Text Spliting")
    texts = text_splitter.split_documents(documents)
    print("Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    print("DB FAISS")
    db = FAISS.from_documents(texts, embeddings)
    print("DB FAISS")
    db.save_local(DB_FAISS_PATH)
    print("Function End")

if __name__ == "__main__":
    print("Calling the function")
    create_vector_db()

