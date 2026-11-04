import numpy as np

from hybrid_retriever import HybridRetriever

# 1. Sahte Embedding Modeli (Test İçin)

class MockEmbedModel:
    def get_text_embedding(self, text: str) -> list:
        # Metnin karakter toplamına göre sabit (deterministik) tohum üretiyoruz
        seed_val = sum(ord(char) for char in text) % 1000
        rng = np.random.default_rng(seed_val)
        
        # Generator nesnesinden 4 elemanlı rastgele dizi üretip listeye çeviriyoruz
        vector = rng.random(4).tolist()
        return vector

# 2. Sahte Chunk Verileri (Semantic Chunker Çıktısı Gibi)
mock_chunks = [
    {
        "chunk_id": 0,
        "text": "Kazan sıcaklığı 120 C seviyesini aşmamalıdır. Basınç 3 bar olmalıdır.",
        "sentence_count": 2
    },
    {
        "chunk_id": 1,
        "text": "Sistemde ERR-404-P aşırı ısınma hatası alındığında soğutma vanası açılır.",
        "sentence_count": 1
    },
    {
        "chunk_id": 2,
        "text": "Yıllık bakım talimatı: Motor yağ değişimi ve filtre kontrolü yapılmalıdır.",
        "sentence_count": 1
    }
]

# 3. TESTİ BAŞLATMA
if __name__ == "__main__":
    print("=== HYBRID RETRIEVER TESTİ BAŞLIYOR ===\n")
    
    embed_model = MockEmbedModel()
    retriever = HybridRetriever(chunks=mock_chunks, embed_model=embed_model, rrf_k=60)
    
    # Test Sorgusu
    query = "ERR-404-P ısınma hatası"
    print(f"\nSorgu: '{query}'\n")
    
    # A) Sadece Dense Arama Testi
    print("--- 1. Dense (Vektör) Arama Sonuçları ---")
    dense_res = retriever.search_dense(query, top_k=2)
    for res in dense_res:
        print(f"Chunk ID: {res['chunk']['chunk_id']} | Dense Skor: {res['dense_score']:.4f}")
        print(f"Metin: {res['chunk']['text']}\n")

    # B) Sadece Sparse (BM25) Arama Testi
    print("--- 2. Sparse (BM25) Arama Sonuçları ---")
    sparse_res = retriever.search_sparse(query, top_k=2)
    for res in sparse_res:
        print(f"Chunk ID: {res['chunk']['chunk_id']} | BM25 Skor: {res['sparse_score']:.4f}")
        print(f"Metin: {res['chunk']['text']}\n")

    # C) Hibrit (RRF Füzyon) Arama Testi
    print("--- 3. Hibrit (RRF Füzyon) Arama Sonuçları ---")
    hybrid_res = retriever.retrieve(query, top_k=2)
    for res in hybrid_res:
        print(f"Chunk ID: {res['chunk']['chunk_id']} | RRF Skor: {res['rrf_score']:.5f}")
        print(f"Dense Sırada: {res['dense_rank']} | Sparse Sırada: {res['sparse_rank']}")
        print(f"Metin: {res['chunk']['text']}\n")