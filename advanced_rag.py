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
    
    
    #Sistem çalıştığı an gerekli tüm ağır modelleri belleğe yükler.
    #Her soru sorulduğunda modelleri tekrar tekerar yüklemek yerine self. ile sınıfın hafızasına
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
    
    def query (self, user_query: str , top_k_retrieval: int = 6 , top_k_rerank : int = 2):
        """
        Kullanıcı sorusunu alır:
        1. Hybrid Search ile adayları getirir.
        2. Reranker ile en kaliteli parçaları süzer
        3. Ollama LLM ile yanıt üretir.
        """
        if not self.retriever:
            raise ValueError("Lütfen önce doküman yükleyin")
        
        print(f" \n Kullanıcı sorusu : '{user_query}'")
        
        #1. Arama(hybrid search)
        print("BM25 ve Dense Vektör araması yapılıyor.")
        candidate_results = self.retriever.retrieve(user_query, top_k = top_k_retrieval)
        
        #2. Yeniden Sıralama(cross encoder reranking)
        print("Cross encoder ile adaylar süzülüyor.")
        reranked_results = self.reranker.rerank(user_query, candidate_results=candidate_results, top_k=top_k_rerank)
        
        #3. Bağlam oluşturma
        context_blocks = []
        for i, res in enumerate(reranked_results):
            chunk_text = res["chunk"]["text"]
            score = res["rerank_score"]
            context_blocks.append(f"--- BAĞLAM PARÇASI #{i+1} (Alakalılık Skoru: {score:.4f}) ---\n{chunk_text}")    
            
        formatted_context = "\n\n".join(context_blocks)
        
        # 4. Strict RAG Prompt Hazırlama
        prompt = f"""Sen teknik dokümanlara göre kesin ve doğru cevaplar veren uzman bir yapay zeka asistansın.
            Sadece sana verilen aşağıdaki BAĞLAM bilgilerini kullanarak kullanıcının sorusuna cevap ver.
            Eğer verilen bağlamda sorunun cevabı YOKSA, kesinlikle kendi bilginden uydurma yapma ve 'Verilen dokümanda bu sorunun cevabı bulunmamaktadır.' de.

            BAĞLAM:
            {formatted_context}

            KULLANICI SORUSU:
            {user_query}

            CEVAP:"""

    # 5. Ollama ile Yanıt Üretme
        print("🧠 [LLM Generation] Ollama yanıt üretiyor...")
        response = self.llm.complete(prompt)
            
        return {
                "query": user_query,
                "answer": str(response),
                "retrieved_chunks": reranked_results,
                "prompt_used": prompt
            }


# === TEST KODU ===
if __name__ == "__main__":
    pdf_path = "data/manual.pdf"
    
    # 1. Boru hattını başlat
    pipeline = AdvancedRAGPipeline(ollama_model_name="llama3.2")
    
    # 2. Dokümanı yükle ve indeksle
    pipeline.ingest_document(pdf_path)
    
    # 3. Örnek bir soru sor
    query_text = "Kazan sıcaklığı kaç dereceyi geçmemelidir?"
    result = pipeline.query(query_text, top_k_retrieval=5, top_k_rerank=2)
    
    print("\n" + "=" * 60)
    print("🤖 OLLAMA LLM CEVABI:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    
    print("\n📌 KULLANILAN EN ALAKALI PARÇALAR (RERANKED):")
    for item in result["retrieved_chunks"]:
        print(f"• [Rerank Skor: {item['rerank_score']:.4f}] -> \"{item['chunk']['text'][:120]}...\"")
