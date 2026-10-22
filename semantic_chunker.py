#Dokümandan gelen ham metni cümlelere ayıracak.
#bge-small-en-v1.5 ile her cümlenin embedding vektörünü alacak.
#ardisik cümleler arasındaki anlamsal mesafeyi hesaplayacak.

import os
import re
import numpy as np
from typing import List, Dict, Any

#HuggingFace ve tokenizer uyarılarını bastırma
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGING_FACE_HUB_TOKEN"]=""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

class SemanticChunker:
    
    #Constructor -> Bu sınıftan nesne üretildigi an otomatik olarak bu fonksiyon calıstirilacak.
    #Bu parçalayıcı hangi embedding modeli kullansın, anlamsal kopuşları yakalarken ne kadar hassas davransın ? 
    def __init__(self, embed_model: HuggingFaceEmbedding, threshold_percentile: float = 85.0):
        self.embed_model = embed_model
        self.threshold_percentile = threshold_percentile
    
    
    #Ham PDF metnini terimleri ortadan ikiye bölmeden parçalar.
    #Yardımcı fonksiyon, baska bir sınıftan cagrılmaz.
    #Regex ile teknik kodları buluyoruz.
    def _split_into_sentences(self, text: str) -> List[str]:
        """Metni noktalama işaretleri ve satır sonlarına göre temiz cümlelere ayırır.
        Teknik kodların bölünmesini engeller.
        """
        raw_sentences = re.split(r'(?<=[.?!])\s+|\n{2,}', text)
        sentences = []
        for s in raw_sentences:
            cleaned = s.strip()
            # Çok kısa/anlamsız karakter kırıntılarını ele
            if len(cleaned) > 10:
                sentences.append(cleaned)
        return sentences
    
    
    #Embedding modelinden çıkan 2 cümlenin vektörlerini alıp anlamsal uzaklıklarını hesaplıyoruz.
    def _cosine_distance(self, v1: List[float], v2: List[float]) -> float:
        """
        İki vektör arasındaki kosinüs mesafesini hesaplar:
        Mesafe = 1 - CosineSimilarity
        (Değer 0'a yakınsa anlamsal olarak aynı konu, 1 e yakınsa konu tamamen farklı)
        """
        
        a = np.array(v1)
        b = np.array(v2)
        
        norm_product = np.linalg.norm(a)*np.linalg.norm(b) + 1e-8
        similarity = np.dot(a, b) / norm_product
        
        return float(1.0 - similarity)        
        
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Metni alır, cümle vektörlerini çıkarır, mesafeleri hesaplar ve eşik değerini aşan noktalardan chunk olusturur.
        """
        sentences = self._split_into_sentences(text)
        print (f" Dokumandan toplam {len(sentences)} adet mantiksal cumle cikarildi.")
        
        if len(sentences) <= 1: #eger metinde sadece 0 ya da 1 cümle varsa bölünecek ve karşılaştırılacak bir sey yoktur.
            return [{"chunk_id":0, "text": text, "sentence_count": len(sentences)}] #sistem hesaplama yapmadan cümleyi döner.
        
        #Cümlelerin BGE embedding vektörlerini üret
        
        print("Cumle embeddingleri hesaplaniyor...")
        embeddings = [self.embed_model.get_text_embedding(s) for s in sentences]
        
        #Ardisik cümleler arasindaki kosinüs mesafesini ölç
        distances = []
        for i in range(len(embeddings)-1):
            dist = self._cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)
            
        #Istatistiksel anlamsal kopus esigini belirle.
        threshold = float(np.percentile(distances, self.threshold_percentile))
        mean_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))
        
        print("Mesafe Istatistikleri: ")
        print(f"Ortalama mesafe: {mean_dist: .4f}")
        print(f"Maksimum Mesafe : {max_dist: .4f}")
        print(f"Esik degeri :(%{self.threshold_percentile}) : {threshold:.4f} ")
        
        #Esigi asan noktalardan parcaları bol.
        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_idx = 0
        
        for i, dist in enumerate(distances):
            if dist > threshold:
                #Anlamsal kopus gerceklesti.Mevcut parcayi kaydet
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_id" : chunk_idx,
                    "text" : chunk_text,
                    "sentence_count": len(current_chunk_sentences),
                    "split_distance" : dist
                }) 
                
                chunk_idx +=1
                current_chunk_sentences = [sentences[i + 1]]
            else:
                current_chunk_sentences.append(sentences[i + 1])  
                
        #Kalan son cümle grubunu ekle 
        if current_chunk_sentences:
            chunks.append({
                "chunk_id" : chunk_idx,
                "text" : " ".join(current_chunk_sentences),
                "sentence_count" : len(current_chunk_sentences),
                "split_distance" : None
            }) 
            
        return chunks
    
    
if __name__ == "__main__":
    pdf_path = "data/manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ {pdf_path} dosyası bulunamadı! Lütfen yolu kontrol edin.")
        exit(1)

    print("📄 PyMuPDF ile dosya okunuyor...")
    reader = PyMuPDFReader()
    docs = reader.load(file_path=pdf_path)
    full_text = "\n".join([doc.text for doc in docs])

    print("🚀 BGE Embedding modeli yükleniyor...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", embed_batch_size=10)

    print("\n⚡ Semantic Chunker test ediliyor...")
    chunker = SemanticChunker(embed_model=embed_model, threshold_percentile=85.0)
    chunks = chunker.create_chunks(full_text)

    print(f"\n✅ Toplam {len(chunks)} adet anlamsal parça (chunk) üretildi!\n")
    print("=" * 60)
    for c in chunks:
        print(f"🔹 [CHUNK #{c['chunk_id']}] (Cümle Sayısı: {c['sentence_count']})")
        if c['split_distance'] is not None:
            print(f"   Kopuş Mesafesi: {c['split_distance']:.4f}")
        print(f"   İçerik: \"{c['text'][:160]}...\"\n")
    print("=" * 60)
    
    
            
                