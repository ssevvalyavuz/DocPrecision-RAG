# DocPrecision-RAG

**DocPrecision-RAG**, endüstriyel teknik dokümantasyon (`data/X200_Kapsamli_Teknik_Kilavuz.pdf`) üzerinde klasik **Naive RAG** yaklaşımı ile **Semantic Chunking + Hibrit Arama (BM25 + Dense) + Reciprocal Rank Fusion (RRF)** tabanlı **Advanced RAG** mimarisini karşılaştıran bir Retrieval-Augmented Generation (RAG) mühendislik ve değerlendirme projesidir[cite: 5, 7, 10, 12].

**Ragas 0.4.3 Collections API** ile (`Top-3`, `n=30` alan-spesifik soru, `retrieval_only` modu) gerçekleştirilen kıyaslamada Advanced RAG boru hattı, **Context Recall** skorunu **0.9667** seviyesinde tutarken **Context Precision** değerini **+%17.57** (`0.8222` → `0.9667`) artırmıştır[cite: 7]. *(Not: Bu sürümde harici bir Cross-Encoder yeniden sıralama modeli kullanılmamış; sıralama kalitesi doğrudan RRF füzyon puanları üzerinden sağlanmıştır[cite: 5, 12].)*

---

## 📊 Retrieval Benchmark Sonuçları (Ragas 0.4.3 Collections)

![DocPrecision-RAG Benchmark](benchmark_comparison.png)

| Metrik (`Top-3`, `n=30`) | Naive RAG Baseline | Advanced RAG Pipeline | Mutlak Fark | Göreli Değişim |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision** | `0.8222` | **`0.9667`** | `+0.1445` | **+%17.57** |
| **Context Recall** | `0.9500` | **`0.9667`** | `+0.0167` | **+%1.76** |
| **Ort. Retrieval Süresi** | **`0.4075 sn`** | `1.2354 sn` | `+0.8279 sn` | `3.03x` ödünleşim |

### Temel Mühendislik Çıkarımları

* **Cross-Encoder Olmadan Yüksek Sıralama Kesinliği (+%17.57 Context Precision):** Ek bir Cross-Encoder gecikmesi eklenmeden yalnızca **Semantic Chunking** (`threshold_percentile=85.0`) ve **RRF (`rrf_k=60`)** tabanlı hibrit füzyon kullanılarak doğru teknik bağlamın ilk sırada getirilme oranı `0.8222`'den `0.9667`'ye yükselmiştir[cite: 5, 12].
* **Tablo ve Teknik Kod Hassasiyeti:** `pdfplumber` ile sayfalardaki tabloları Markdown formatında ayrıştıran ingest katmanı[cite: 5] ve kelime nadirliğini (IDF) koruyan `BM25Okapi` motoru[cite: 9, 12], `E-309`, `E-412` gibi arıza kodları ile basınç/sıcaklık eşiklerini yüksek isabetle eşleştirir[cite: 10].
* **Gecikme ve Doğruluk Ödünleşimi (Latency Trade-off):** Çift yönlü indeks taraması (Dense `BAAI/bge-m3` + Sparse `BM25Okapi`) ve RRF hesaplaması sorgu başına ortalama `+0.8279 sn` ek süre (`0.4075 sn` → `1.2354 sn`) getirirken bağlam gürültüsünü belirgin şekilde azaltır[cite: 5, 12].

---

## 🏗️ Sistem Mimarisi

```text
                       ┌──────────────────────────────────────────────┐
                       │   Teknik PDF Kılavuzu (X-200 Bakım Kılavuzu) │
                       └──────────────────────┬───────────────────────┘
                                              │
                ┌─────────────────────────────┴─────────────────────────────┐
                ▼                                                           ▼
┌───────────────────────────────────┐                       ┌───────────────────────────────────┐
│        Naive RAG Baseline         │                       │       Advanced RAG Pipeline       │
├───────────────────────────────────┤                       ├───────────────────────────────────┤
│ 1. Sabit Boyutlu Karakter Bölme   │                       │ 1. pdfplumber Tablo & Metin Okuma │
│ 2. Dense Vektör Gömme             │                       │ 2. Semantic Chunking (%85 eşik)   │
│ 3. Kosinüs Benzerliği Sıralama    │                       │ 3. Hibrit Arama (BM25 + BAAI/bge-m3)
│                                   │                       │ 4. Reciprocal Rank Fusion (k=60)  │
└─────────────────┬─────────────────┘                       └─────────────────┬─────────────────┘
                  │ Top-3 Parça                                               │ Top-3 Parça
                  └────────────────────────────┬──────────────────────────────┘
                                               ▼
                       ┌──────────────────────────────────────────────┐
                       │   Ragas 0.4.3 Collections Değerlendirme      │
                       │   • ContextPrecision (Sıralama Kesinliği)    │
                       │   • ContextRecall (Referans Kapsama Oranı)   │
                       └──────────────────────────────────────────────┘
```

### Bileşen Karşılaştırması

| Katman | Naive RAG Baseline (`naive_rag.py`) | Advanced RAG Pipeline (`advanced_rag.py`) |
| --- | --- | --- |
| **Doküman Ayrıştırma** | Düz metin okuma | `pdfplumber` ile sayfa bazlı tablo |
| **Bölümleme (Chunking)** | Sabit boyutlu karakter bölme | `SemanticChunker` (`threshold_percentile=85.0`) |
| **Gömme Modeli (Embedding)** | Dense embedding | `BAAI/bge-m3` (`HuggingFaceEmbedding`) |
| **Arama ve Sıralama** | Doğrudan kosinüs benzerliği (`top_k=3`) | `HybridRetriever`: `BM25Okapi` + Dense kosinüs benzerliği + RRF (`rrf_k=60`, `top_k=6` aday havuzu) |
| **Reranking** | Yok | Harici Cross-Encoder yok; doğrudan RRF füzyon skoru sıralaması |
| **Yanıt Üretimi (UI Modu)** | — | OpenAI uyumlu protokol üzerinden Groq Cloud API (`OpenAILike`) |

---

## 📂 Proje Yapısı

```text
DocPrecision-RAG/
├── benchmark_comparison.png            # Retrieval & gecikme karşılaştırma grafiği
├── data/
│   └── X200_Kapsamli_Teknik_Kilavuz.pdf # Sentetik endüstriyel kazan teknik kılavuzu[cite: 7, 10]
├── evaluation/
│   ├── questions.json                  # ground_truth içeren 30 soruluk değerlendirme veri seti[cite: 7]
│   └── ragas_collections_results.json  # Soru bazlı ve özet Ragas 0.4.3 metrik sonuçları[cite: 7]
├── naive_rag.py                        # Temel dense vektör retrieval boru hattı[cite: 7]
├── advanced_rag.py                     # Semantic chunking + Hibrit RRF boru hattı[cite: 5]
├── semantic_chunker.py                 # Anlamsal sınır tespit ve parçalama modülü[cite: 5]
├── hybrid_retriever.py                 # BM25Okapi + Dense vektör + RRF arama motoru[cite: 12]
├── app.py                              # Streamlit tabanlı etkileşimli PDF soru-cevap arayüzü[cite: 6]
├── benchmark_2.py                      # Checkpoint korumalı Ragas 0.4.3 değerlendirme betiği[cite: 7]
├── groq_utils.py                       # HTTP 429 hız sınırı için üstel geri çekilme (retry) yardımcıları[cite: 11]
├── generate_detailed_pdf_manual.py     # Türkçe font destekli X-200 teknik kılavuz PDF üreticisi[cite: 10]
├── .gitignore
└── README.md

```

---

## ⚙️ Kurulum ve Yapılandırma

### 1. Depoyu Klonlayın ve Sanal Ortam Oluşturun

```bash
git clone https://github.com/ssevvalyavuz/DocPrecision-RAG.git
cd DocPrecision-RAG

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

```

### 2. Bağımlılıkları Yükleyin

```bash
pip install streamlit ragas openai groq llama-index llama-index-llms-openai-like \
           llama-index-embeddings-huggingface pdfplumber rank-bm25 reportlab \
           python-dotenv numpy

```

### 3. Ortam Değişkenlerini Tanımlayın

Proje kök dizininde `.env` dosyası oluşturun:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

```

(Güvenlik Notu: `.env` dosyası `.gitignore` içinde tanımlıdır; API anahtarlarınızı repoya göndermeyin.)

---

## 🚀 Kullanım

### 1. Streamlit Web Arayüzünü Başlatma (`app.py`)

Teknik bir PDF dokümanı yüklemek, anlamsal parçalama sayısını görmek ve seçilen bağlam parçalarını RRF skorlarıyla birlikte incelemek için:

```bash
streamlit run app.py

```

### 2. Ragas 0.4.3 Collections Benchmark Çalıştırma (`benchmark_2.py`)

Her iki boru hattını LLM yanıt üretimi kapalı (`retrieval_only`) modda 30 soru üzerinden kıyaslamak için:

```bash
python benchmark_2.py

```

* **Kesinti Korumalı Checkpoint:** Her sorunun sonucu `evaluation/ragas_collections_checkpoint.json` dosyasına atomik olarak yazılır; olası API limit kesintilerinde betik kaldığı sorudan devam eder.

* **Nihai Çıktı:** Tüm sorular tamamlandığında özet ve detay skorlar `evaluation/ragas_collections_results.json` dosyasına kaydedilir.



### 3. Örnek Teknik Kılavuz PDF'ini Yeniden Üretme

`data/X200_Kapsamli_Teknik_Kilavuz.pdf` dosyasını sıfırdan üretmek için:

```bash
python generate_detailed_pdf_manual.py

```

### GitHub'a Gönderme Komutu

*(Eğer `benchmark_comparison.png` dosyasını `assets/` klasörüne koyduysan yukarıdaki 2 yerde `assets/benchmark_comparison.png` olarak güncellemen yeterlidir.)*

```bash
git add benchmark_comparison.png README.md evaluation/questions.json evaluation/ragas_collections_results.json
git commit -m "docs: add benchmark chart and updated README"
git push

```