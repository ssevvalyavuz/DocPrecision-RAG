import re
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """
    Dense (Vektör / Kosinüs Benzerliği) ve Sparse (BM25) aramalarını 
    RRF (Reciprocal Rank Fusion) ile birleştiren Hibrit Arama Motoru.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], embed_model: Any, rrf_k : int = 60):
        """
        Retriever'ı chunk listesi ve embedding modeli ile başlatır.
        """
        self.chunks = chunks
        self.embed_model = embed_model
        self.rrf_k = rrf_k
        
        #BM25 İndeksi Kurulumu 
        
        #Parçalanmış kelime torbalarını toplayacağımız boş bir liste açıyoruz.
        self.corpus_tokenized=[]
        
        #Her bir chunk'ı sırayla elimize alıyoruz.
        for chunk in self.chunks:
            #Chunk sözlüğünün içindeki düz metni çekiyoruz.
            raw_text = chunk["text"] #text anahtarının karsisindaki metni alır.
            
            #metni _tokenize fonksiyonuna gönderip kelime listesine çeviriyoruz.
            tokens = self._tokenize(raw_text)
            
            self.corpus_tokenized.append(tokens)
        
        self.bm25 = BM25Okapi(self.corpus_tokenized)
        
        #Chunk vektörlerinin önceden hesaplanması:
        #List Comprehension
        #Cebimizdeki self.chunk listesindeki her bir chunk ı al içindeki text i bulup embedini al. Çıkan tüm bu vektörleri yeni bir liste yapıp self.chunk_embeddings e kaydet.
        print (f"Retriever: {len(self.chunks)} chunk için embedding indekslemesi yapılıyor..."),
        self.chunk_embeddings = [
            self.embed_model.get_text_embedding(chunk["text"])
            for chunk in self.chunks
        ]
        
    def _tokenize (self, text:str) -> List[str]:
            """
            BM25 için metni küçük harfe çevirip kelimelere ayırır
            """

            cleaned = text.lower()
            tokens = re.findall(r"\b[\w\-]+\b", cleaned)
            return tokens
        
    def _cosine_similarity (self, v1: List[float], v2: List[float]) -> float:
            """İki embedding arasındaki kosinüs benzerliğini hesaplar
            """
            a = np.array(v1)
            b = np.array(v2)
            
            product = (np.linalg.norm(a)*np.linalg.norm(b)) + 1e-8
            return float (np.dot(a,b)/product)
        
    def search_dense(self, query: str, top_k: int = 5) -> List[Dict[str,Any]]:
        """Kullanıcı sorgusunun vektörel benzerlik aramasını yapar.
        
        """
        query_embedding = self.embed_model.get_text_embedding(query)
        scored_chunks = []
        
        idx = 0 
        
        for chunk_emb in self.chunk_embeddings:
            score = self._cosine_similarity(query_embedding, chunk_emb) #Kullanıcının sorusu ile o anki chunk vektörü arasındaki Kosinüs benzerliği

            current_chunk = self.chunks[idx]
            
            chunk_info = {
                "chunk" : current_chunk,
                "dense_score" : score,
                "original_idx" : idx  
            }
            
            scored_chunks.append(chunk_info)
            
            idx = idx +1 
            
            # Skora göre büyükten küçüğe sırala
        scored_chunks.sort(key=lambda x: x["dense_score"], reverse=True)
        return scored_chunks[:top_k]
    
    
    def search_sparse (self, query: str, top_k: int = 5 ) -> List[Dict[str, Any]]:
        """ BM25 kullanarak metinsel (anahtar kelime) arama yapar.
        """
        
        #Kullanıcının sorduğu düz cümleyi alıp temiz kelimelerine ayırıyoruz.
        
        query_tokens = self._tokenize(query)
        
        # BM25 motoruna bu kelimeleri verip her chunk için bir uygunluk skoru alıyoruz.
        scores = self.bm25.get_scores(query_tokens)
        
        #Puanlanmış chunk kartlarını biriktireceğimiz boş ana listeyi açıyoruz
        scored_chunks = []
        
        idx = 0
        
        #BM25 motorunun döndürdüğü skorlar listesini tek tek dönüyoruz.
        
        for score in scores:
            # o anki indekse karsılık gelen orijinal ham metin chunkını çekiyoruz.
            current_chunk = self.chunks[idx]
            
            numeric_score = float(score)
            
            chunk_info = {
                "chunk" : current_chunk ,
                "sparse_score" : numeric_score,
                "original_idx" : idx
            }
            
            scored_chunks.append(chunk_info)
            
            idx = idx + 1
            
        #Sıralama işlemi
        
        #    Açık sıralama fonksiyonumuz için bir yardımcı sıralama anahtarı fonksiyonu tanımlıyoruz
        def get_sparse_score(item):
            return item["sparse_score"]

        #    Listeyi 'sparse_score' değerine göre büyükten küçüğe (reverse=True) sıralıyoruz
        scored_chunks.sort(key=get_sparse_score, reverse=True)

        # 7. Kullanıcının istediği adet kadar (top_k, örn: en iyi 5 tane) sonucu dilimleyip döndürüyoruz
        top_results = scored_chunks[:top_k]

        return top_results
            
        
        
        
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """ Ana Hibrit Fonksiyon : Dense ve Sparse aramalarını çalıştırır.
        RRF ile füzyon skorunu hesaplar
        """
        
        #Her iki kanaldan da aday havuzunu topla
        dense_results = self.search_dense(query, top_k=len(self.chunks))
        sparse_results = self.search_sparse(query, top_k=len(self.chunks))
        
        # RRF Puan Havuzu: {chunk_id: {"chunk": ..., "rrf_score": float, "dense_rank": int, "sparse_rank": int}}
        rrf_scores: Dict[int, dict[str,Any]] = {}
        
        rank = 1
        
        #Dense sıralama
        for item in dense_results:
            idx = item ["original_idx"]
            
            #rrf formülüne göre bu sıradaki parça için puan hesaplıyoruz.
            
            rrf_point = 1.0 / (self.rrf_k + rank)
            
            #Eğer bu parça (idx) daha önce ortak rrf_scores sözlüğüne EKLENMEMİŞSE:
            #Sözlükte bu parça için sıfırdan bir kayıt oluşturuyoruz.
            
            if idx not in rrf_scores:
                rrf_scores[idx] = {
                    "chunk" : item["chunk"],
                    "rrf_score" : 0.0, #Puanı sıfırdan başlatıyoruz.
                    "dense_rank" : rank,
                    "sparse_rank" : None, #sparse sırası henüz belli değil.
                    "dense_score" : item["dense_score"] #orijinal benzerlik skoru
                }
            
            #Hesapladığımız RRF puanını bu parçanın mevcut RRF puanına ekliyoruz.
            
            rrf_scores[idx]["rrf_score"] = rrf_scores[idx]["rrf_score"] + rrf_point
            
            rank = rank + 1
            
            #Sparse sıralama
            
        rank = 1
            
            #BM25 aramasından gelen sıralı sonuçları tek tek ele alıyoruz.
        for item in sparse_results:
                idx = item["original_idx"]
                
                # b) RRF formülüne göre bu BM25 sırasındaki parça için puan hesaplıyoruz: 
                #Puan = 1 / (k + derece)
                rrf_point = 1.0 / (self.rrf_k + rank)
                
                #KRİTİK ADIM 
                #EĞER bu parça (idx) az önceki Dnese arama döngüsünde 'rrf_scores' sözlüğüne EKLENMİŞSE:
                
                if idx in rrf_scores:
                    #Mevcut RRF puanının üzerine BM25 ten gelen RRF puanını ekliyoruz (FÜZYON/BİRLEŞTİRME):
                    rrf_scores[idx]["rrf_score"] = rrf_scores[idx]["rrf_score"] + rrf_point
                    
                    #Eksik duran sparse sıralamasını kaydediyoruz
                    rrf_scores[idx]["sparse_rank"] = rank
                    
                    rrf_scores[idx]["sparse_score"] = item["sparse_score"]
                    
                #DEĞİLSE(yani bu parça vektör aramasında ilk sonuçlara giremediyse, Sadece bm25 te yakalandıysa)
                
                else:
                    #Sözlükte bu parça için sıfırdan yepyeni bir kayıt açıyoruz.
                    
                    rrf_scores[idx] = {
                        "chunk": item["chunk"],
                        "rrf_score" : rrf_point, #Doğrudan bm25 rrf puanıyla başlatyıyoruz.
                        "dense_rank" : None, #Dense tarafında listeye giremediği için None.
                        "sparse_score" : item["sparse_score"] #Orijinal bm25 skoru. 
                    }
                    
                rank = rank + 1 
                    
        final_ranked = sorted(
                    rrf_scores.values(), 
                    key=lambda x: x["rrf_score"], 
                    reverse=True
                )

        return final_ranked[:top_k]
                        
                
                
                
            
            
        
        
        
        
        
        
            
            
