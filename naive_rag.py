import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import fitz  # PyMuPDF
from groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NaiveRAGPipeline:
    """
    DocPrecision-RAG için Naive RAG Baseline Boru Hattı.
    Gelişmiş bileşenler (Semantic Chunking, BM25, RRF, Cross-Encoder) çıkarılmıştır.
    """
    
    def __init__(
        self, 
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        model_name: str = "openai/gpt-oss-20b",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        print("🚀 Naive RAG Baseline Boru Hattı Başlatılıyor...")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(".env dosyasında GROQ_API_KEY bulunamadı!")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        print(f"📦 Embedding modeli yükleniyor: {embed_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, embed_batch_size=4)
        
        print("⚡ Groq API İstemcisi Başlatılıyor...")
        self.client = Groq(api_key=groq_api_key)
        
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_embeddings: List[List[float]] = []

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """Metni sabit karakter boyutunda ve örtüşmeli (overlap) parçalara böler."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks

    def _cosine_similarity(self, query_vector: List[float], doc_vectors: List[List[float]]) -> np.ndarray:
        """Sorgu ile doküman vektörleri arasındaki Cosine Similarity değerlerini hesaplar."""
        q_vec = np.array(query_vector)
        d_vecs = np.array(doc_vectors)
        
        norm_q = np.linalg.norm(q_vec)
        norm_d = np.linalg.norm(d_vecs, axis=1)
        
        # Sıfıra bölünme koruması
        norm_d[norm_d == 0] = 1e-10
        if norm_q == 0:
            norm_q = 1e-10

        sims = np.dot(d_vecs, q_vec) / (norm_d * norm_q)
        return sims

    def ingest_document(self, pdf_path: str) -> None:
        """Dokümanı okur, fixed-size chunk'lar üretir ve vektör indeksini oluşturur."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {pdf_path}")
            
        print(f"📄 Naive PDF Okuma başlatılıyor: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text("text") for page in doc])
        doc.close()

        raw_chunks = self._fixed_size_chunking(full_text)
        
        self.chunks = []
        self.chunk_embeddings = []
        
        print(f"✂️ {len(raw_chunks)} adet sabit boyutlu (Size: {self.chunk_size}, Overlap: {self.chunk_overlap}) chunk oluşturuldu.")
        
        for idx, text_block in enumerate(raw_chunks):
            emb = self.embed_model.get_text_embedding(text_block)
            self.chunks.append({
                "chunk_id": idx,
                "text": text_block
            })
            self.chunk_embeddings.append(emb)

        print("✅ Naive Vektör İndeksi Tamamlandı.")

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Naive RAG sorgusunu çalıştırır ve detaylı metrik çıktısı üretir."""
        if not self.chunks or not self.chunk_embeddings:
            raise ValueError("Lütfen önce bir doküman indeksleyin!")

        start_total = time.time()
        
        # 1. Retrieval Aşaması
        start_retrieval = time.time()
        query_embedding = self.embed_model.get_query_embedding(user_query)
        similarities = self._cosine_similarity(query_embedding, self.chunk_embeddings)
        
        # Top-K İndeks Seçimi
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved_chunks_output = []
        context_blocks = []
        
        for rank, idx in enumerate(top_indices):
            chunk_data = self.chunks[idx]
            score = float(similarities[idx])
            
            context_blocks.append(f"--- BAĞLAM PARÇASI #{rank+1} ---\n{chunk_data['text']}")
            retrieved_chunks_output.append({
                "chunk_id": chunk_data["chunk_id"],
                "text": chunk_data["text"],
                "similarity_score": score
            })
            
        retrieval_latency = time.time() - start_retrieval
        
        # 2. Generation Aşaması (Advanced RAG ile Birebir Aynı Prompt)
        start_gen = time.time()
        formatted_context = "\n\n".join(context_blocks)
        
        prompt = f"""Aşağıdaki bağlam bilgisini dikkatlice oku ve soruyu bu bilgilere dayanarak Türkçe yanıtla. 
Cevabı 2-3 cümlede kısa ve öz ver.

BAĞLAM:
{formatted_context}

SORU:
{user_query}

CEVAP:"""

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        
        answer_text = chat_completion.choices[0].message.content
        generation_latency = time.time() - start_gen
        total_latency = time.time() - start_total
        
        return {
            "question": user_query,
            "retrieved_chunks": retrieved_chunks_output,
            "answer": answer_text,
            "retrieval_latency": round(retrieval_latency, 4),
            "generation_latency": round(generation_latency, 4),
            "total_latency": round(total_latency, 4)
        }

if __name__ == "__main__":
    # data/ klasörü altındaki PDF dosya yolunu belirtiyoruz
    test_pdf = os.path.join("data", "X200_Kapsamli_Teknik_Kilavuz.pdf") 
    
    if os.path.exists(test_pdf):
        # 1. Pipeline'ı başlat
        pipeline = NaiveRAGPipeline()
        
        # 2. PDF'i oku ve fixed-size chunking yap
        pipeline.ingest_document(test_pdf)
        
        # 3. Test sorusu sor
        soru = "Kazan taş ve tortu temizliği (Descaling) hangi periyotta ve kimin sorumluluğunda yapılmalıdır?"
        res = pipeline.query(soru, top_k=3)
        
        # 4. Programatik çıktıyı kontrol et
        print("\n--- NAIVE RAG ÇIKTI KONTROLÜ ---")
        print(f"Soru: {res['question']}")
        print(f"Cevap: {res['answer']}\n")
        print(f"Getirilen Chunk Sayısı: {len(res['retrieved_chunks'])}")
        print(f"Retrieval Süresi: {res['retrieval_latency']} saniye")
        print(f"Generation Süresi: {res['generation_latency']} saniye")
        print(f"Toplam Süre: {res['total_latency']} saniye")
    else:
        print(f"❌ Test için '{test_pdf}' bulunamadı. Lütfen data/ altındaki dosya adını kontrol et.")