# 🚀 DocPrecision-RAG: Gelişmiş RAG Mimarisi

> **Karşılaştırmalı RAG Motoru:** Naive RAG temel yaklaşımından; Semantic Chunking, Hybrid Search (BM25 + BGE) ve Cross-Encoder Reranking kullanan gelişmiş RAG mimarisi.

**DocPrecision-RAG**, sabit karakter sayısına dayalı standart (Naive) RAG yaklaşımlarının ötesine geçerek; metni anlamsal mesafe kopuşlarına göre bölen (**Semantic Chunking**), kelime tabanlı ve vektörel aramayı RRF ile birleştiren (**Hybrid Search**) ve bulunan sonuçları Cross-Encoder ile yeniden sıralayan (**Reranking**) uçtan uca, modüler bir RAG mimarisidir.

---

## 🏛️ Sistem Mimarisi

```text
                               [ Ham PDF Dokümanı ]
                                        │
                                        ▼
                              [ PyMuPDF Doküman Okuyucu ]
                                        │
                                        ▼
                              [ Özel Semantic Chunker ]
                  (Dinamik Percentile Tabanlı Cosine Distance Threshold)
                                        │
                                        ▼
                              [ Hibrit Arama İndeksi ]
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
   [ BM25 Retriever ]                                       [ BGE Dense Retriever ]
 (Kesin Kelime Eşleşmesi)                                (BAAI/bge-small-en-v1.5)
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        ▼
                       [ Reciprocal Rank Fusion (RRF) ]
                              (Birleşik Aday Havuzu)
                                        │
                                        ▼
                          [ Cross-Encoder Reranker ]
                          (BAAI/bge-reranker-base)
                                        │
                                        ▼
                           [ Katı Prompt Mühendisliği ]
                                        │
                                        ▼
                             [ Yerel LLM (Ollama) ]
                               (Llama 3.2 / Mistral)
```

---

## 🔑 Temel Özellikler ve Teknik Detaylar

- **Semantic Chunking (`semantic_chunker.py`):** Sabit boyutlu Naive Chunking yaklaşımı yerine, `BAAI/bge-small-en-v1.5` kullanarak cümleler arasındaki anlamsal mesafeyi analiz eder. Dinamik eşik belirleme yöntemiyle metni anlamsal geçiş noktalarından böler. Bu projede eşik değeri **%85'lik percentile** üzerinden belirlenmektedir.

- **Hybrid Retrieval Engine (`hybrid_retriever.py`):** Kesin terimler, model ID'leri ve hata kodları gibi ifadeler için **BM25**, bağlamsal sorgular için ise **Dense BGE Embeddings** kullanır. İki arama sonucunu **Reciprocal Rank Fusion (RRF)** ile birleştirir. RRF parametresi `k=60` olarak kullanılmaktadır.

- **Cross-Encoder Reranking (`reranker.py`):** `BAAI/bge-reranker-base` modeli ile sorgu ve doküman parçalarını birlikte değerlendirir. Böylece düşük alakalı içerikleri filtreleyerek LLM'e yalnızca daha ilgili bağlamı gönderir.

- **Hallüsinasyon Önleme (`advanced_rag.py`):** LLM'in yalnızca getirilen bağlamı kullanmasını sağlayan katı prompt kuralları uygular. Yeterli bilgi bulunmadığında sistemin açık bir bilgi bulunamadı cevabı üretmesini sağlar.

---

## 🛠️ Proje Yapısı

```text
DocPrecision-RAG/
├── data/
│   └── manual.pdf             # Örnek teknik PDF dokümanı
├── semantic_chunker.py        # Cosine distance tabanlı semantic chunking
├── hybrid_retriever.py        # BM25 + Dense Vector + RRF retrieval modülü
├── reranker.py                # Cross-Encoder reranking modülü
├── advanced_rag.py            # Uçtan uca RAG pipeline yöneticisi
├── requirements.txt           # Python bağımlılıkları
└── README.md                  # Proje dokümantasyonu
```

---

## 🚀 Hızlı Başlangıç

### 1. Ön Gereksinimler

Sistemi çalıştırmak için aşağıdakilerin kurulu olması gerekir:

- Python 3.10+
- Ollama
- Git

Ollama üzerinden kullanılacak yerel LLM modeli de indirilmelidir.

Örneğin:

```bash
ollama pull llama3.2
```

---

### 2. Projeyi Klonlama

```bash
git clone https://github.com/ssevvalyavuz/DocPrecision-RAG.git
cd DocPrecision-RAG
```

---

### 3. Sanal Ortam Oluşturma

```bash
python3 -m venv .venv
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\activate
```

---

### 4. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

---

### 5. Ollama Modelini Çalıştırma

Llama 3.2 kullanmak için:

```bash
ollama pull llama3.2
```

Alternatif olarak Mistral kullanılabilir:

```bash
ollama pull mistral
```

---

### 6. Gelişmiş RAG Pipeline'ını Çalıştırma

```bash
python advanced_rag.py
```

Pipeline çalıştırıldığında sistem sırasıyla:

1. PDF dokümanını okur.
2. Metni PyMuPDF ile çıkarır.
3. Metni anlamsal olarak parçalara böler.
4. BGE embedding'lerini oluşturur.
5. BM25 ve Dense Retrieval indekslerini oluşturur.
6. Kullanıcı sorgusunu iki farklı yöntemle arar.
7. Sonuçları RRF ile birleştirir.
8. Cross-Encoder ile yeniden sıralar.
9. En alakalı içerikleri prompt içerisine yerleştirir.
10. Prompt'u yerel Ollama LLM'ine gönderir.
11. Nihai cevabı üretir.

---

## 📊 RAG Pipeline'ı

DocPrecision-RAG aşağıdaki çok aşamalı mimariyi kullanır:

```text
PDF Dokümanı
     │
     ▼
PyMuPDF ile Metin Çıkarma
     │
     ▼
Semantic Chunking
     │
     ▼
┌───────────────────────┐
│                       │
│    Hibrit Retrieval   │
│                       │
├───────────┬───────────┤
│           │           │
▼           ▼           │
BM25      BGE Dense     │
│           │           │
└─────┬─────┘           │
      ▼                 │
     RRF ◄──────────────┘
      │
      ▼
Birleşik Aday Havuzu
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
En Alakalı Context
      │
      ▼
Strict Prompt
      │
      ▼
Yerel LLM
      │
      ▼
Nihai Cevap
```

---

## 🧠 Semantic Chunking

Semantic Chunking, dokümanı sabit karakter veya token sayısına göre bölmek yerine, metindeki anlamsal geçiş noktalarını tespit ederek parçalamayı amaçlar.

### Naive Chunking

Geleneksel yaklaşımda:

```text
Her 500 karakter → Yeni Chunk
```

gibi sabit bir bölme yöntemi kullanılabilir.

Bu yaklaşımda aynı konuya ait iki cümlenin farklı chunk'lara ayrılması veya farklı konuların aynı chunk içerisinde bulunması mümkündür.

### DocPrecision-RAG Yaklaşımı

Bu projede cümlelerin embedding'leri oluşturulur ve ardışık cümleler arasındaki cosine distance hesaplanır.

```text
Cümle
  │
  ▼
Embedding
  │
  ▼
Cosine Distance
  │
  ▼
Mesafe Dağılımı
  │
  ▼
%85 Percentile Eşik Değeri
  │
  ▼
Anlamsal Geçiş Noktası
  │
  ▼
Yeni Chunk
```

Kullanılan embedding modeli:

```text
BAAI/bge-small-en-v1.5
```

Bu sayede chunk sınırları dokümanın anlamsal yapısına daha fazla uyum sağlar.

### Semantic Chunking'in Avantajları

- Konu bütünlüğünü daha iyi korur.
- İlgili cümlelerin aynı chunk içerisinde kalmasını sağlar.
- Anlamsal geçişlerde daha doğru sınırlar oluşturur.
- Sabit chunk boyutuna olan bağımlılığı azaltır.
- Retrieval kalitesinin artırılmasına yardımcı olur.

---

## 🔎 Hybrid Search

Hybrid Search, iki farklı retrieval yaklaşımını birlikte kullanır:

1. **BM25 — Kelime tabanlı retrieval**
2. **Dense Retrieval — Anlamsal retrieval**

Bu iki yaklaşımın güçlü yönleri RRF ile birleştirilir.

---

### 🔤 BM25 Retrieval

BM25, sorgudaki kelimelerin dokümanlarda bulunmasına dayalı bir retrieval yöntemidir.

Özellikle aşağıdaki sorgu tiplerinde avantajlıdır:

- Model isimleri
- Hata kodları
- Ürün kodları
- Teknik terimler
- API isimleri
- Değişken isimleri
- Kesin anahtar kelimeler

Örneğin:

```text
ERR_CONNECTION_REFUSED
API_KEY
SQLSTATE 42P01
Model X100
```

gibi ifadelerde BM25 güçlü sonuçlar sağlayabilir.

---

### 🧠 Dense Retrieval

Dense Retrieval, metinleri embedding vektörlerine dönüştürerek anlamsal benzerlik üzerinden arama yapar.

Kullanılan model:

```text
BAAI/bge-small-en-v1.5
```

Örneğin kullanıcı:

```text
Uygulama neden veritabanına bağlanamıyor?
```

diye sorduğunda, dokümanda tam olarak bu cümle bulunmasa bile bağlantı problemiyle ilgili anlamsal olarak benzer içerikler bulunabilir.

---

### 🔗 Hibrit Retrieval Akışı

```text
                 Kullanıcı Sorgusu
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         BM25 Arama          BGE Dense Arama
              │                   │
              ▼                   ▼
        Kelime Sıralaması    Anlamsal Sıralama
              │                   │
              └─────────┬─────────┘
                        ▼
                       RRF
                        │
                        ▼
              Birleşik Sıralama
```

---

## 🔗 Reciprocal Rank Fusion (RRF)

BM25 ve Dense Retrieval farklı skor sistemleri kullandığından, sonuçları doğrudan karşılaştırmak yerine sıralamalarını birleştirmek için **Reciprocal Rank Fusion** kullanılır.

Genel formül:

$$
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
$$

Burada:

- $d$ = doküman
- $R$ = retrieval sistemleri
- $rank_r(d)$ = dokümanın ilgili retrieval sistemindeki sırası
- $k$ = sıralama sabiti

Bu projede:

$$
k = 60
$$

olarak kullanılmaktadır.

RRF sayesinde BM25 ve Dense Retrieval sonuçları ortak bir sıralama altında birleştirilir.

---

## 🎯 Cross-Encoder Reranking

İlk retrieval aşaması yüksek recall elde etmeye odaklanırken, Cross-Encoder ikinci aşamada daha hassas bir relevance değerlendirmesi gerçekleştirir.

Kullanılan model:

```text
BAAI/bge-reranker-base
```

Pipeline:

```text
Çok Sayıda Chunk
       │
       ▼
İlk Retrieval
       │
       ▼
Aday Chunk'lar
       │
       ▼
Cross-Encoder
       │
       ▼
Alaka Skorlaması
       │
       ▼
En İlgili Chunk'lar
       │
       ▼
LLM
```

Cross-Encoder, sorguyu ve doküman parçasını birlikte değerlendirerek daha hassas bir relevance skoru üretir.

Böylece LLM'e gereksiz veya düşük alakalı içerik gönderilmesinin azaltılması amaçlanır.

---

## 🛡️ Hallüsinasyon Önleme

RAG sistemlerinde yalnızca doğru dokümanı getirmek yeterli değildir. LLM'in getirilen bağlam dışındaki bilgileri kullanarak cevap üretmesi de engellenmelidir.

DocPrecision-RAG bu nedenle **Strict Prompt Engineering** yaklaşımı kullanır.

Temel mantık:

```text
Retrieved Context
       │
       ▼
Gerekli bilgi context içerisinde var mı?
       │
      ┌┴┐
      │ │
     EVET HAYIR
      │ │
      ▼ ▼
    Cevap  Fallback
    üret   üret
```

Context içerisinde yeterli bilgi bulunmuyorsa sistem açık bir fallback cevabı döndürür.

Örneğin:

```text
Information not found in the provided context.
```

Bu yaklaşım, LLM'in desteklenmeyen bilgiler üretmesini azaltmayı amaçlar.

---

## 🧩 Proje Yapısı

```text
DocPrecision-RAG/
│
├── data/
│   └── manual.pdf
│
├── semantic_chunker.py
│   └── Semantic Chunking motoru
│
├── hybrid_retriever.py
│   └── BM25 + Dense Retrieval + RRF
│
├── reranker.py
│   └── Cross-Encoder Reranking
│
├── advanced_rag.py
│   └── Uçtan uca RAG pipeline
│
├── requirements.txt
│   └── Python bağımlılıkları
│
└── README.md
    └── Proje dokümantasyonu
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| Python | Ana geliştirme dili |
| PyMuPDF | PDF metin çıkarma |
| BAAI/bge-small-en-v1.5 | Dense embedding |
| BM25 | Kelime tabanlı retrieval |
| RRF | Retrieval sonuçlarını birleştirme |
| BAAI/bge-reranker-base | Cross-Encoder reranking |
| Ollama | Yerel LLM çalıştırma |
| Llama 3.2 | Yerel LLM |
| Mistral | Alternatif yerel LLM |

---

## 🆚 Naive RAG ve Advanced RAG Karşılaştırması

| Özellik | Naive RAG | DocPrecision-RAG |
|---|---|---|
| Doküman Okuma | PDF Reader | PyMuPDF |
| Chunking | Sabit boyutlu | Semantic Chunking |
| Embedding | Dense | BGE Dense Embedding |
| Keyword Search | ❌ | ✅ BM25 |
| Semantic Search | ✅ | ✅ |
| Hybrid Search | ❌ | ✅ |
| Rank Fusion | ❌ | ✅ RRF |
| Reranking | ❌ | ✅ Cross-Encoder |
| Hallüsinasyon Kontrolü | Temel | Strict Prompting |
| LLM | API / Local | Local Ollama |
| Mimari | Basit | Çok Aşamalı ve Modüler |

---

## 🔬 Neden Advanced RAG?

Naive RAG genellikle aşağıdaki yapıyı kullanır:

```text
PDF
 ↓
Sabit Boyutlu Chunking
 ↓
Vector Embedding
 ↓
Vector Search
 ↓
LLM
```

Bu yaklaşım basit ve hızlı olsa da aşağıdaki problemlere sahip olabilir:

- Chunk sınırları anlamsal bütünlüğü bozabilir.
- Kesin anahtar kelimeler semantic search tarafından kaçırılabilir.
- Anlamsal olarak benzer ancak alakasız içerikler retrieval sonucuna girebilir.
- Gereksiz context LLM'in cevap kalitesini düşürebilir.
- Fazla context gürültü oluşturabilir.
- Yetersiz retrieval durumunda LLM desteklenmeyen bilgiler üretebilir.

DocPrecision-RAG ise retrieval sürecini birden fazla aşamaya ayırır:

```text
PDF
 ↓
Semantic Chunking
 ↓
┌──────────────────────┐
│                      │
│   Hybrid Retrieval   │
│                      │
├──────────┬───────────┤
│          │           │
▼          ▼           │
BM25    Dense BGE      │
│          │           │
└────┬─────┘           │
     ▼                 │
    RRF                 │
     │                  │
     ▼                  │
Cross-Encoder Reranker │
     │                  │
     ▼                  │
Strict Prompt           │
     │                  │
     ▼                  │
Local LLM               │
```

---

## 🔄 Uçtan Uca İş Akışı

```text
                    PDF
                     │
                     ▼
              PyMuPDF Reader
                     │
                     ▼
            Cümlelerin Çıkarılması
                     │
                     ▼
             Semantic Chunking
                     │
                     ▼
             ┌───────┴───────┐
             ▼               ▼
           BM25          BGE Embedding
             │               │
             ▼               ▼
      Kelime Sıralaması  Dense Sıralama
             │               │
             └───────┬───────┘
                     ▼
                    RRF
                     │
                     ▼
             Aday Chunk'lar
                     │
                     ▼
          Cross-Encoder Reranker
                     │
                     ▼
              En İlgili Context
                     │
                     ▼
              Strict Prompt
                     │
                     ▼
                Ollama LLM
                     │
                     ▼
              Nihai Cevap
```

---

## 🧪 Örnek Sorgu Akışı

Kullanıcının şu soruyu sorduğunu düşünelim:

```text
Uygulama neden veritabanı bağlantı hatası veriyor?
```

### 1. BM25

Sorgudaki anahtar kelimeler üzerinden arama yapılır:

```text
veritabanı
bağlantı
hata
connection
database
timeout
```

### 2. Dense Retrieval

Sorgunun anlamsal karşılığı aranır.

Dokümanda tam olarak:

```text
Uygulama neden veritabanı bağlantı hatası veriyor?
```

ifadesi bulunmasa bile aynı problemi açıklayan içerikler bulunabilir.

### 3. RRF

BM25 ve Dense Retrieval sıralamaları birleştirilir.

### 4. Cross-Encoder

Aday chunk'lar sorguyla birlikte değerlendirilir.

En alakalı içerikler üst sıralara taşınır.

### 5. Strict Prompt

Yalnızca seçilen context LLM'e gönderilir.

### 6. LLM

LLM, retrieved context'e dayanarak nihai cevabı oluşturur.

---

## 📈 Retrieval Stratejisi

DocPrecision-RAG retrieval sürecini iki aşamalı olarak ele alır:

### Birinci Aşama — Candidate Retrieval

Amaç mümkün olduğunca ilgili adayları yakalamaktır.

```text
BM25
  +
Dense Retrieval
  ↓
RRF
  ↓
Candidate Pool
```

### İkinci Aşama — Precision Reranking

Amaç adaylar içerisinden gerçekten en alakalı içerikleri seçmektir.

```text
Candidate Pool
      ↓
Cross-Encoder
      ↓
Relevance Score
      ↓
Top Chunks
```

Bu yapı, hızlı retrieval ile daha hassas relevance değerlendirmesini bir araya getirir.

---

## 📊 Değerlendirme Stratejisi

Sistemin performansını değerlendirmek için Naive RAG ve Advanced RAG sonuçları karşılaştırılabilir.

### Naive RAG

```text
Sabit Chunking
      ↓
Dense Retrieval
      ↓
LLM
```

### Advanced RAG

```text
Semantic Chunking
      ↓
BM25 + Dense Retrieval
      ↓
RRF
      ↓
Cross-Encoder Reranking
      ↓
Strict Prompt
      ↓
LLM
```

### Kullanılabilecek Metrikler

- Retrieval Precision
- Retrieval Recall
- Precision@K
- Recall@K
- MRR
- NDCG
- Context Relevance
- Answer Relevance
- Answer Faithfulness
- Hallüsinasyon Oranı
- Retrieval Süresi
- Toplam Yanıt Süresi

---

## 📌 Gelecek Geliştirmeler

- [ ] Naive RAG baseline oluşturulması
- [ ] Naive RAG ve Advanced RAG karşılaştırma script'i
- [ ] Recall@K ve Precision@K benchmark'ları
- [ ] MRR / NDCG değerlendirmesi
- [ ] Farklı embedding modellerinin karşılaştırılması
- [ ] Farklı reranker modellerinin karşılaştırılması
- [ ] Query Expansion eklenmesi
- [ ] Query Rewriting eklenmesi
- [ ] Metadata filtering eklenmesi
- [ ] Doküman kaynak/citation gösterimi
- [ ] Conversational Memory eklenmesi
- [ ] Multi-document retrieval
- [ ] FastAPI backend
- [ ] Web arayüzü
- [ ] Docker desteği
- [ ] Logging ve monitoring
- [ ] Latency benchmark
- [ ] Token kullanım analizi
- [ ] Otomatik RAG değerlendirme sistemi

---

## 🧭 Geliştirme Yol Haritası

### Faz 1 — Temel RAG

- [x] PDF ingestion
- [x] Semantic Chunking
- [x] Dense Retrieval
- [x] BM25 Retrieval
- [x] RRF
- [x] Cross-Encoder Reranking
- [x] Local LLM entegrasyonu

### Faz 2 — Değerlendirme

- [ ] Naive RAG baseline
- [ ] Retrieval benchmark
- [ ] Reranking benchmark
- [ ] Hallüsinasyon değerlendirmesi
- [ ] Latency karşılaştırması
- [ ] Retrieval kalitesi ölçümü

### Faz 3 — Production

- [ ] FastAPI API katmanı
- [ ] Web arayüzü
- [ ] Docker
- [ ] Logging
- [ ] Monitoring
- [ ] Configuration yönetimi

### Faz 4 — Gelişmiş Retrieval

- [ ] Query Rewriting
- [ ] Multi-Query Retrieval
- [ ] Parent Document Retrieval
- [ ] Metadata-aware Retrieval
- [ ] Multi-document reasoning

---

## 🎓 Projede Gösterilen Temel Kavramlar

Bu proje aşağıdaki modern RAG ve yapay zekâ kavramlarının pratik bir uygulamasını göstermektedir:

- **Retrieval-Augmented Generation (RAG)**
- **Semantic Chunking**
- **Dense Vector Retrieval**
- **Lexical Retrieval**
- **BM25**
- **Hybrid Search**
- **Reciprocal Rank Fusion (RRF)**
- **Cross-Encoder Reranking**
- **Prompt Engineering**
- **Hallüsinasyon Önleme**
- **Local LLM**
- **Ollama**
- **Çok Aşamalı Retrieval**
- **Modüler RAG Mimarisi**

---

## 👩‍💻 Geliştirici

**Şevval Yavuz**

Bilgisayar Mühendisliği

GitHub: [@ssevvalyavuz](https://github.com/ssevvalyavuz)

---

## 📄 Lisans

Bu proje eğitim, araştırma ve portföy amaçlı geliştirilmiştir.