from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from .vector_store import get_vector_store
# We no longer need the query_transformer for this simplified version
# from .query_transformer import create_multi_query_retriever 

def create_retrieval_engine(top_k=10, rerank_top_n=3):
    """
    Creates a retrieval engine with re-ranking (Multi-query is temporarily disabled).
    """
    vector_store = get_vector_store()
    
    # 1. Create the base retriever directly from the vector store
    base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # NOTE: The multi_query_retriever step is skipped to stay within free API limits.
    # multi_query_retriever = create_multi_query_retriever(base_retriever)

    # 2. Initialize the cross-encoder model for re-ranking
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 3. Create the re-ranker compressor
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=rerank_top_n)
    
    # 4. Create the final Contextual Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever # IMPORTANT: Use the base_retriever here now
    )
    
    return compression_retriever