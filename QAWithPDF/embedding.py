from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        #service_context = ServiceContext.from_defaults(llm=model,embed_model=gemini_embed_model, chunk_size=800, chunk_overlap=20)
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.num_output = 512
        Settings.context_window = 3900
        Settings.transformations = [SentenceSplitter(chunk_size=1024)]
        
        logging.info("")
        #index = VectorStoreIndex.from_documents(document,service_context=service_context)
        index = VectorStoreIndex.from_documents(
        document, embed_model=gemini_embed_model, transformations=Settings.transformations)
        index.storage_context.persist()
        
        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise customexception(e,sys)