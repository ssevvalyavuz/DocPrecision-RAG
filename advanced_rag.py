import os 
from typing import List, Dict, Any
from dotenv import load_dotenv

# LlamaIndex Groq uyumlu istemci entegrasyonu

load_dotenv()

from llama_index.llms.groq import Groq
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
    4. Groq LLM Generation
    """
    
    
    #Sistem çalıştığı an gerekli tüm ağır modelleri belleğe yükler.
    #Her soru sorulduğunda modelleri tekrar tekerar yüklemek yerine self. ile sınıfın hafızasına
    import os 
from typing import List, Dict, Any
from dotenv import load_dotenv

# load_dotenv'i modül yüklenirken en üstte çağırıyoruz
load_dotenv()

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

from semantic_chunker import SemanticChunker
from hybrid_retriever import HybridRetriever 

os.environ["TOKENIZERS_PARALLELISIM"] = "false"

class AdvancedRAGPipeline:
    
    import os 
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Groq import'u YERİNE doğrudan OpenAILike import ediyoruz
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

from semantic_chunker import SemanticChunker
from hybrid_retriever import HybridRetriever 

os.environ["TOKENIZERS_PARALLELISIM"] = "false"

class AdvancedRAGPipeline:
    def __init__(
        self, 
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        groq_model_name: str = "openai/gpt-oss-20b"
    ):
        print("Advanced RAG Boru Hattı Başlatılıyor...")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(".env dosyasında GROQ_API_KEY bulunamadı!")

        groq_model_name = os.getenv("GROQ_MODEL", groq_model_name)

        print(f"{embed_model_name} modeli yükleniyor...")
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, embed_batch_size=10)
        
        print(f"Groq LLM bağlantısı kuruluyor (model: {groq_model_name})...")
        
        # OpenAILike uses the OpenAI-compatible protocol with Groq's endpoint fixed here.
        self.llm = OpenAILike(
            model=groq_model_name,
            api_base="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            is_chat_model=True
        )        
        
        self.chunker = SemanticChunker(embed_model=self.embed_model, threshold_percentile=85.0)
        self.retriever = None
        self.chunks = []
        
    def ingest_document(self, pdf_path:str):
        """
        PDF dokümanını okur, anlamsal olarak parçalar ve hibrit arama indeksini kurar.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("Dosya bulunamadi !")
        print("Doküman okunuyor....")
        reader = PyMuPDFReader()
        docs = reader.load(file_path=pdf_path)
        full_text = "\n". join([doc.text for doc in docs])
        
        print("\n---ADIM 1: SEMANTIC CHUNKER---")
        self.chunks = self.chunker.create_chunks(full_text)
        print(f"Toplam {len(self.chunks)} adet anlamsal chunk üretildi...")
        
        print("\n---ADIM 2: HYBRID ARAMA (BM25 +BGE)---")
        self.retriever = HybridRetriever(chunks=self.chunks, embed_model=self.embed_model , rrf_k=60)
        print("Hibrit arama indeksi hazır.")
    
    def query(self, user_query: str, top_k_retrieval: int = 6, top_k_rerank: int = 2):
        """
        Kullanıcı sorusunu alır:
        1. Hybrid Search ile adayları getirir.
        2. Reranker ile en kaliteli parçaları süzer
        3. Groq LLM ile yanıt üretir.
        """
        if not self.retriever:
            raise ValueError("Lütfen önce doküman yükleyin")
        
        print(f"\nKullanıcı sorusu : '{user_query}'")
        
        # 1. Arama: Hibrit arama ile top_k_retrieval kadar aday getirilir (Örn: 6 parça)
        print("BM25 ve Dense Vektör araması yapılıyor...")
        candidate_results = self.retriever.retrieve(user_query, top_k=top_k_retrieval)
        
        # 2. Seçim: Gelen adaylar arasından top_k_rerank kadarı LLM'e verilmek üzere süzülür (Örn: En iyi 2 parça)
        selected_results = candidate_results[:top_k_rerank]
        
        # 3. Bağlam oluşturma
        context_blocks = []
        retrieved_chunks_output = []
        
        for i, res in enumerate(selected_results):
            chunk_text = res.node.get_content() if hasattr(res, "node") else str(res)
            score = getattr(res, "score", 0.0)

            context_blocks.append(f"--- BAĞLAM PARÇASI #{i+1} ---\n{chunk_text}")
            
            retrieved_chunks_output.append({
                "chunk": {"text": chunk_text},
                "rerank_score": score
            })
            
        formatted_context = "\n\n".join(context_blocks)
        
           
        # 4. Strict RAG Prompt Hazırlama
        prompt = f"""Aşağıdaki bağlam bilgisini dikkatlice oku ve soruyu bu bilgilere dayanarak Türkçe yanıtla. 
        Cevabı 2-3 cümlede kısa ve öz ver.

        BAĞLAM:
        {formatted_context}

        SORU:
        {user_query}

        CEVAP:"""
    # 5. Groq ile yanıt üretme
        print("🧠 [LLM Generation] Groq yanıt üretiyor...")
        response = self.llm.complete(prompt)
            
        return {
                "query": user_query,
                "answer": str(response),
                "retrieved_chunks": retrieved_chunks_output,
                "prompt_used": prompt
            }

