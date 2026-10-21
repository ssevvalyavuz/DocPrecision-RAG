## FAZ 1: Akıllı Veri Hazırlığı (Semantic Ingestion Pipeline)

Geleneksel (Naive) RAG sistemlerinin aksine metin, sabit karakter sınırlarıyla körlemesine bölünmez. Veri hazırlığı anlamsal bütünlük korunarak işletilir.

### 1. Temiz Metin Çıkarma (Parsing)
* **Bileşen:** `PyMuPDF (fitz)`
* **İşlev:** PDF'in sıkıştırılmış bayt akışlarını (`FlateDecode`) ve sayfa yerleşim koordinatlarını çözer. Font ve biçimlendirme çöplerinden arındırılmış, salt metin katmanını belleğe aktarır.

### 2. Vektörel Mesafe Tabanlı Parçalama (Semantic Chunking)
* **Model:** `BAAI/bge-small-en-v1.5` (Transformer tabanlı Bi-Encoder)
* **Çalışma Prensibi:**
  1. Doküman mantıksal cümle dizilerine ayrılır ($S_1, S_2, S_3, \dots$).
  2. Her bir cümle modelden geçirilerek 384 boyutlu embedding vektörleri ($E_i$) elde edilir.
  3. Ardışık cümleler arasındaki kosinüs mesafesi hesaplanır:
     $$\text{Mesafe} = 1 - \frac{E_i \cdot E_{i+1}}{\Vert{}E_i\Vert{} \Vert{}E_{i+1}\Vert{}}$$
  4. Hesaplanan mesafeler istatistiksel bir eşik değerle (örneğin %85'lik persentil) karşılaştırılır. Eşiği aşan noktalarda konu değişimi tespit edilir ve yeni bir parça (chunk) başlatılır.
* **Kazanım:** Başlık, arıza kodu ve çözüm adımlarının ortadan ikiye bölünmesi engellenir.

---

## FAZ 2: İki Kademeli Hibrit Arama ve Üretim (Two-Stage Retrieval & Generation)

Sorgu aşamasında tek bir arama yöntemi yerine, birbirinin açığını kapatan hibrit bir arama ve hakem modelleme mimarisi çalışır.

### 1. Aşama: Kaba Toplama (Coarse Retrieval)
Sorgu iki paralel arama hattından geçirilir:
* **Dense Retrieval (BGE Embedding):** Sorunun anlamsal/kavramsal niyetini analiz eder (ör. "kazanın durması" ile "basınç valfi arızası" eşleşmesi).
* **Sparse Retrieval (BM25):** Terim frekansı (TF) ve ters belge frekansı (IDF) hesaplarıyla birebir eşleşmelere odaklanır (`ERR-404-P`, `J-3`, `4.2 bar`).
* **Çıktı:** İki kaynaktan toplanan en yüksek skorlu ilk 10 aday parça havuza alınır.

### 2. Aşama: Hassas Yeniden Sıralama (Fine Reranking)
* **Model:** `BAAI/bge-reranker-base` (Cross-Encoder)
* **İşlev:** Bi-Encoder modelleri soru ve dokümanı bağımsız vektörleştirdiği için aralarındaki ince bağlamsal etkileşimleri kaçırabilir. Cross-Encoder, soru ile aday parçayı aynı anda tek bir Transformer katmanına sokarak çapraz dikkat (cross-attention) hesaplar:
  $$[CLS] + \text{Soru} + [SEP] + \text{Aday Parça} + [SEP]$$
* **Çıktı:** Havuzdaki 10 parça arasından soruya doğrudan yanıt teşkil eden en yüksek ilgililiğe sahip **ilk 2 parça** seçilir.

### 3. Aşama: Bağlamı Zenginleştirilmiş Üretim (Augmented Generation)
* **Model:** `Ollama (Llama 3.2)`
* **İşlev:** Cross-Encoder tarafından elenen 2 parça, katı bir sistem prompt'u eşliğinde modele iletilir. Model, `temperature=0.1` seviyesinde çalıştırılarak sadece kanıt metinlerdeki teknik bilgileri aktarır, halüsinasyon üretimi engellenir.