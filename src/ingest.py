import json
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class QAPair(BaseModel):
    id: str
    category: str
    question: str
    answer: str

def load_qa_pairs(path: str) -> list[QAPair]:
    """
    Loads the JSON
    Parses each entry into a QAPair (hint: QAPair(**item) for each item in the loaded list)
    Checks for duplicate ids — raise ValueError if found, with a message naming which ones
    Checks there are at least 30 entries — raise ValueError (or just a print warning, your call) if not
    Returns the list of QAPair objects
    """
    with open(path, 'r') as f:
        data = json.load(f)

    qa_pairs = [QAPair(**item) for item in data]

    # Check for duplicate ids
    ids = [qa.id for qa in qa_pairs]
    duplicates = set([id for id in ids if ids.count(id) > 1])
    if duplicates:
        raise ValueError(f"Duplicate IDs found: {', '.join(duplicates)}")

    # Check for at least 30 entries
    if len(qa_pairs) < 30:
        print("Warning: Less than 30 entries found.")

    return qa_pairs 

## Checkpoint: in Terminal, run the following to test the QA pair loading:
# python -c "
# from src.ingest import load_qa_pairs, build_documents
# pairs = load_qa_pairs('data/kb/qa_pairs.json')
# docs = build_documents(pairs)
# print(f'{len(docs)} documents built')
# print(docs[0].page_content)
# print(docs[0].metadata)
# "


def build_documents(pairs: list[QAPair]) -> list[Document]:

    """
    Convert QAPair objects into LangChain Documents for embedding.

    page_content combines question + answer so retrieval can match on
    either phrasing, and the answer travels with the retrieved result.
    metadata keeps id/category/question/answer as separate structured
    fields for filtering and display without re-parsing page_content.
    """

    documents = []
    for pair in pairs:
        page_content = f"Question: {pair.question}\nAnswer: {pair.answer}"
        metadata = {
            "id": pair.id,
            "category": pair.category,
            "question": pair.question,
            "answer": pair.answer
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    return documents
    


def get_embedding_function():
    """Returns the local embedding model used to convert text into vectors.
    Uses sentence-transformers/all-MiniLM-L6-v2 — small, free, runs locally."""
    
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(documents: list[Document], persist_directory: str = "./chroma_db") -> Chroma:
    """Embeds documents and stores them in a persistent Chroma vector store on disk.
    Uses each document's id (from metadata) as its Chroma id, so re-running
    ingestion on the same persist_directory updates existing entries in place
    instead of creating duplicates..
    """


    embedding_function = get_embedding_function()
    # Extract IDs from document metadata
    ids = [doc.metadata["id"] for doc in documents]
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory,
        ids=ids
    )
    
    return vectorstore

## Checkpoint: in Terminal, run the following to test the ingestion pipeline and vector store:
# python -c "
# from src.ingest import load_qa_pairs, build_documents, build_vectorstore
# pairs = load_qa_pairs('data/kb/qa_pairs.json')
# docs = build_documents(pairs)
# vectorstore = build_vectorstore(docs)
# print('Vector store built with', vectorstore._collection.count(), 'entries')
# results = vectorstore.similarity_search('how do I get a monitor for home', k=2)
# for r in results:
#     print('-', r.metadata['id'], ':', r.metadata['question'])
# "

