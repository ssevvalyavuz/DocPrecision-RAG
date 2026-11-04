from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """ Hybrid_retriever dan gelen aday parçaları Cross-Encoder modeli ile soru-metin
    bütünlüğü üzerinden yeniden puanlayıp sıralayan Reranker sınıfı
    
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base" ):
        """Cross encoder modelini hugging-face üzerinden yükler
        """
        
        print(f"Reranker : '{model_name}' modeli yükleniyor")
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, candidate_results: List[Dict[str, Any]], top_k: int = 3) -> List:
        """Kullanıcı sorusu ile aday parçaları kıyaslar, yeniden puanlar ve top_k kadarki en alakalı parçayı döndürür
        
        """
        
        if not candidate_results:
            return[]
        
        #Modele vermek için çifleri buluyoruz.
        pairs = []
        for item in candidate_results:
            chunk_text = item ["chunk"]["text"]
            pairs.append([query,chunk_text])
            
        #Cross encoder modeli ile çifleri yeniden puanlıyoruz.
        
        raw_scores = self.model.predict(pairs)
        
        reranked_results = []
        idx = 0
        
        for item in candidate_results:
            #skor bi numpy elemanı olarak da dönebileceği için float a çeviriyoruz.
            score = float(raw_scores[idx])
            
            reranked_item =  {
                "chunk": item["chunk"],
                "rerank_score": score,
                "rrf_score": item.get("rrf_score"),
                "dense_rank": item.get("dense_rank"),
                "sparse_rank": item.get("sparse_rank")
            }
            
            reranked_results.append(reranked_item)
            idx = idx +1
            
        #Sonuçları yeni 'rerank_score' değerine göre büyükten küçüğe sıralıyoruz.
        # Elimizdeki listede sadece düz sayılar yok; her eleman içinde metin, RRF puanı, ID ve rerank_score bulunan karmaşık bir sözlüktür.Python'a "Bu sözlük kartlarının neresine bakarak sıralama yapacağını" söylememiz gerekir.
        def get_rerank_score(item):
            return item ["rerank_score"]
        
        reranked_results.sort(key = get_rerank_score, reverse=True)
        
        return reranked_results [:top_k ] #Sıralanmış listeden sadece en baştaki top_k elemanı al.
            
        
        