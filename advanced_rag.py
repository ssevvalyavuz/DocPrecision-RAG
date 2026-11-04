import os 
from typing import List, Dict, Any

#LlamaIndex Ollama entegrasyonu

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

from semantic_chunker import SemanticChunker
from hybrid_retriever import HybridRetriever 
from reranker import CrossEncoderReranker

os.environ["TOKENIZERS_PARALLELISIM"] = "false"

class AdvancedRAGPipeline:
    """ Advenced RAG boru hattı:
    1. Semantic Chunker
    2. Hybrid Search
    3. Cross-Encoder Reranker
    4. Ollama LLM Generation
    """
    
def __init__(self, embed_model_name : str = "BAAI/bge-small-en-v1.5",reranker_model_name : str = "BAAI/bge-reranker-base",ollama_model_name : str ="llama3.2"):
    print("Advanced RAG Boru Hattı Başlatılıyor...")
    
    #Embedding modeli yükleme
    print(f"{'embed_model_name'} modeli yükleniyor...")
    self.embed_model = HuggingFaceEmbedding (model_name=embed_model_name , embed_batch_size=10)
    
    #Reranker modeli yükleme
    print (f"{'reranker_model_name'} modeli yükleniyor ...")
    self.reranker = CrossEncoderReranker(model_name=reranker_model_name)
    
    #Ollama LLM bağlantısı
    print ("Ollama llm bağlantısı kuruluyor ...")
    self.llm = Ollama(model=ollama_model_name, request_timeout=120.0)
    
    #Parçalayıcı
    self.chunker = SemanticChunker(embed_model=self.embed_model, threshold_percentile=85.0)
    
    self.retriever = None
    self.chunks = []
    
