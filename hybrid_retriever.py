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
        
        self.bm25 = BM250Kapi(self.corpus_tokenized)
        
        
            
            
