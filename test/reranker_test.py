from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    HybridRetriever'dan gelen aday parçaları Cross-Encoder modeli ile
    soru-metin bütünlüğü üzerinden yeniden puanlayıp sıralayan sınıftır.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"Reranker: '{model_name}' modeli yükleniyor...")
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, candidate_results: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        if not candidate_results:
            return []
            
        # 1. [Soru, Metin] çiftlerini hazırlıyoruz
        pairs = []
        for item in candidate_results:
            chunk_text = item["chunk"]["text"]
            pairs.append([query, chunk_text])
            
        # 2. Model ile puan üretiyoruz
        raw_scores = self.model.predict(pairs)
        
        # 3. Skorları parçaların kartlarına zımbalıyoruz
        reranked_results = []
        idx = 0
        for item in candidate_results:
            score = float(raw_scores[idx])
            
            reranked_item = {
                "chunk": item["chunk"],
                "rerank_score": score,
                "rrf_score": item.get("rrf_score")
            }
            reranked_results.append(reranked_item)
            idx = idx + 1
            
        # 4. En yüksek puana göre büyükten küçüğe sıralıyoruz
        def get_rerank_score(item):
            return item["rerank_score"]
            
        reranked_results.sort(key=get_rerank_score, reverse=True)
        
        # 5. En alakalı ilk top_k kadar sonucu döndürüyoruz
        return reranked_results[:top_k]


# === TEST SENARYOSU ===
if __name__ == "__main__":
    print("=== CROSS-ENCODER RERANKER TESTİ BAŞLIYOR ===\n")
    
    # 1. Reranker nesnesini başlatıyoruz (BAAI/bge-reranker-base yüklenecek)
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    
    # 2. Test Sorgusu
    query = "Kazan sıcaklığı kaç dereceyi geçmemelidir?"
    
    # 3. HybridRetriever'dan gelmiş gibi duran Sahte (Mock) Aday Parçalar
    mock_candidates = [
        {
            "chunk": {
                "chunk_id": 0,
                "text": "Yıllık bakım talimatı: Motor yağ değişimi ve filtre kontrolü yapılmalıdır."
            },
            "rrf_score": 0.0322  # RRF'e göre yanlışlıkla ilk sıraya çıkmış bir aday
        },
        {
            "chunk": {
                "chunk_id": 1,
                "text": "Sistemde ERR-404-P hatası alındığında soğutma vanası açılır."
            },
            "rrf_score": 0.0315
        },
        {
            "chunk": {
                "chunk_id": 2,
                "text": "Kazan sıcaklığı 120 C seviyesini aşmamalıdır. Basınç 3 bar olmalıdır."
            },
            "rrf_score": 0.0301  # Doğru cevap ama RRF sıralamasında geride kalmış
        }
    ]
    
    print(f"Sorgu: '{query}'\n")
    print("--- Rerank İşleminden Önce Aday Liste (RRF Sırası) ---")
    for item in mock_candidates:
        print(f"Chunk ID: {item['chunk']['chunk_id']} | RRF Skor: {item['rrf_score']}")
        print(f"Metin: {item['chunk']['text']}\n")
        
    # 4. Rerank Fonksiyonunu Çalıştırıyoruz
    reranked_results = reranker.rerank(query, candidate_results=mock_candidates, top_k=2)
    
    print("=" * 60)
    print("--- Rerank İşleminden Sonra Filtrelenmiş Sonuçlar ---")
    print("=" * 60)
    for item in reranked_results:
        print(f"Chunk ID: {item['chunk']['chunk_id']} | Reranker Skor: {item['rerank_score']:.4f}")
        print(f"Metin: {item['chunk']['text']}\n")