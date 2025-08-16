from .vector_store import get_vector_store
from .query_transformer import create_multi_query_retriever

def create_retrieval_engine():
    vector_store = get_vector_store()
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    multi_query_retriever = create_multi_query_retriever(base_retriever)
    return multi_query_retriever