import os
import re
import json
import math
import time
from typing import List, Dict, Any

# Mevcut pipeline modülleri (Dosyalarına dokunulmaz)
from naive_rag import NaiveRAGPipeline
from advanced_rag import AdvancedRAGPipeline

DATASET_PATH = os.path.join("evaluation", "questions.json")
PDF_PATH = os.path.join("data", "X200_Kapsamli_Teknik_Kilavuz.pdf")
TOP_K = 3

TURKISH_MAP = str.maketrans("çğıöşüİÇĞIÖŞ", "cgiosuicgios")
STOPWORDS = {"ve", "ile", "bir", "bu", "icin", "de", "da", "den", "dan", "kat", "veya", "ise", "olarak"}

# --- 1. TOKEN BAZLI METİN VE KELİME KORUMA YARDIMCILARI ---

def normalize_to_tokens(text: str) -> List[str]:
    """Metni küçük harfe ve İngilizce/TR karaktere normalize eder, noktalama işaretlerinden ayırıp token listesi döner."""
    text = text.lower().translate(TURKISH_MAP)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return tokens

def extract_critical_tokens(text: str) -> List[str]:
    """\b sınırı kullanarak kritik terimleri ve sayısal değerleri tam eşleşme için ayıklar."""
    text_lower = text.lower().translate(TURKISH_MAP)
    patterns = [
        r'\b\d+(?:[\.,]\d+)?\b',  # Tam sayılar ve ondalıklar (Örn: 4,2, 115, 6)
        r'\be-\d+\b',               # Arıza kodları (Örn: e-204)
        r'\bscram\b',               # Özel terimler
        r'\bv-\d+\b',               # Valf kodları (Örn: v-10, v-12)
        r'\bp-\d+\b'                # Pompa kodları (Örn: p-04)
    ]
    tokens = []
    for pat in patterns:
        matches = re.findall(pat, text_lower)
        tokens.extend(matches)
    return list(set(tokens))

# --- 2. GELİŞMİŞ EŞLEŞTİRME VE PARÇALI KONTROL MOTORU ---

def is_chunk_relevant(target_chunk: str, all_k_chunks: List[str], snippet_text: str) -> bool:
    """
    1. Exact Token Eşleşmesi ile tam kelime arar (Sub-string hatası önlenir).
    2. Tam Kelime Sınırı (\b) ile V-1 ve V-12 veya 4 ve 45 ayrımı yapılır.
    3. Parçalı Eşleşme: Chunk %40+ kapsıyorsa VE kalan parça diğer Top-K chunk'larda varsa Relevant=1 sayılır.
    """
    target_tokens = set(normalize_to_tokens(target_chunk))
    snippet_tokens = normalize_to_tokens(snippet_text)

    if not snippet_tokens:
        return False

    snippet_set = set(snippet_tokens)
    matched_tokens = target_tokens.intersection(snippet_set)
    match_ratio = len(matched_tokens) / len(snippet_set)

    # 1. Kritik Terim Kontrolü (Exact Word Boundaries)
    critical_tokens = extract_critical_tokens(snippet_text)
    if critical_tokens:
        chunk_lower = target_chunk.lower().translate(TURKISH_MAP)
        has_all_critical = True
        for ct in critical_tokens:
            pattern = r'\b' + re.escape(ct) + r'\b'
            if not re.search(pattern, chunk_lower):
                has_all_critical = False
                break

        if not has_all_critical:
            return False  # Kritik sayı/kod eksikse direkt elenir (False Positive Engeli)

    # 2. Tek Başına Yeterli Eşleşme (%75+ Token Kapsama)
    if match_ratio >= 0.75:
        return True

    # 3. Context-Aware Parçalı Eşleşme (%40+ Kapsama + Komşu Chunk Tamamlaması)
    if match_ratio >= 0.40:
        other_chunks_combined = " ".join([c for c in all_k_chunks if c != target_chunk])
        other_tokens = set(normalize_to_tokens(other_chunks_combined))
        
        # Snippet'in hedef chunk'ta OLMAYAN eksik kelimeleri
        missing_tokens = snippet_set - target_tokens
        other_matched = other_tokens.intersection(missing_tokens)
        
        # Eksik kalan kelimelerin en az %70'i diğer chunk'larda tamamlanıyorsa geçerli sayılır
        if missing_tokens and (len(other_matched) / len(missing_tokens)) >= 0.70:
            return True

    return False

def check_snippet_covered_in_context(all_k_chunks: List[str], snippet_text: str) -> bool:
    """Recall@K için: Top-K bağlamının tamamı snippet'i kapsıyor mu?"""
    combined_context = " ".join(all_k_chunks)
    combined_tokens = set(normalize_to_tokens(combined_context))
    snippet_tokens = normalize_to_tokens(snippet_text)

    if not snippet_tokens:
        return False

    snippet_set = set(snippet_tokens)
    matched = combined_tokens.intersection(snippet_set)
    match_ratio = len(matched) / len(snippet_set)

    # Kritik terim kontrolü
    critical_tokens = extract_critical_tokens(snippet_text)
    if critical_tokens:
        combined_lower = combined_context.lower().translate(TURKISH_MAP)
        for ct in critical_tokens:
            pattern = r'\b' + re.escape(ct) + r'\b'
            if not re.search(pattern, combined_lower):
                return False

    return match_ratio >= 0.75

# --- 3. METRİK HESAPLAMA MOTORU (DÜZELTİLMİŞ NDCG) ---

def calculate_ndcg(binary_relevance: List[int], total_ideal_snippets: int, k: int) -> float:
    """Ground Truth'taki ideal snippet sayısını temel alan gerçek NDCG@K hesabı."""
    if total_ideal_snippets == 0:
        return 0.0

    # 1. Actual DCG@K
    dcg = 0.0
    for i, rel in enumerate(binary_relevance[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)

    # 2. Ideal DCG@K (Ground Truth'taki ideal relevante göre I vector)
    ideal_ones = min(total_ideal_snippets, k)
    idcg = 0.0
    for i in range(ideal_ones):
        idcg += 1.0 / math.log2(i + 2)

    return (dcg / idcg) if idcg > 0 else 0.0

def evaluate_retrieval(retrieved_chunks: List[str], relevant_snippets: List[str], k: int) -> Dict[str, Any]:
    """Tek Tip Standart Değerlendirme Motoru."""
    if not relevant_snippets:
        return {
            "precision": None,
            "recall": None,
            "mrr": None,
            "ndcg": None,
            "is_na": True
        }

    k_chunks = retrieved_chunks[:k]
    binary_rel = []

    # Chunk Düzeyinde İlgililik (Precision, MRR, NDCG)
    for chunk in k_chunks:
        is_rel = 1 if any(is_chunk_relevant(chunk, k_chunks, snip) for snip in relevant_snippets) else 0
        binary_rel.append(is_rel)

    # Precision@K
    precision = sum(binary_rel) / k if k > 0 else 0.0

    # Recall@K (Snippet Düzeyinde Çift Sayımsız)
    covered_snippets = sum(1 for snip in relevant_snippets if check_snippet_covered_in_context(k_chunks, snip))
    recall = covered_snippets / len(relevant_snippets)

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, rel in enumerate(binary_rel):
        if rel == 1:
            mrr = 1.0 / (rank + 1)
            break

    # NDCG@K (Doğru IDCG Mantığıyla)
    ndcg = calculate_ndcg(binary_rel, total_ideal_snippets=len(relevant_snippets), k=k)

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "is_na": False
    }

# --- 4. BENCHMARK ÇALIŞTIRICI ---

def run_benchmark():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Evaluation verisi bulunamadı: {DATASET_PATH}")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF dosyası bulunamadı: {PDF_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("==================================================")
    print("🚀 DocPrecision-RAG Benchmark Başlatılıyor")
    print(f"📄 PDF: {PDF_PATH}")
    print(f"❓ Soru Sayısı: {len(questions)}")
    print(f"🎯 Top-K: {TOP_K}")
    print("==================================================\n")

    print("--- [1/2] Naive RAG İndeksleniyor ---")
    naive_pipe = NaiveRAGPipeline()
    naive_pipe.ingest_document(PDF_PATH)

    print("\n--- [2/2] Advanced RAG İndeksleniyor ---")
    adv_pipe = AdvancedRAGPipeline()
    adv_pipe.ingest_document(PDF_PATH)

    naive_results = []
    adv_results = []

    print("\n⚡ BENCHMARK EVALUATION ÇALIŞIYOR...\n")

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        snippets = q.get("relevant_snippets", [])

        print(f"❓ [Soru #{q_id}] {q_text}")

        # Naive RAG Çalıştır
        n_res = naive_pipe.query(q_text, top_k=TOP_K)
        n_chunks = [c["text"] for c in n_res["retrieved_chunks"]]
        n_metrics = evaluate_retrieval(n_chunks, snippets, k=TOP_K)
        n_metrics["total_latency"] = n_res["total_latency"]
        naive_results.append(n_metrics)

        # Advanced RAG Çalıştır
        a_res = adv_pipe.query(q_text, top_k_retrieval=6, top_k_rerank=TOP_K)
        a_chunks = [c["chunk"]["text"] for c in a_res["retrieved_chunks"]]
        a_metrics = evaluate_retrieval(a_chunks, snippets, k=TOP_K)
        a_metrics["total_latency"] = a_res.get("total_latency", 0.0)
        adv_results.append(a_metrics)

    # --- ÖZET METRİKLER ---
    def calculate_averages(results: List[Dict[str, Any]]) -> Dict[str, float]:
        valid = [r for r in results if not r["is_na"]]
        if not valid:
            return {}
        return {
            "precision": sum(r["precision"] for r in valid) / len(valid),
            "recall": sum(r["recall"] for r in valid) / len(valid),
            "mrr": sum(r["mrr"] for r in valid) / len(valid),
            "ndcg": sum(r["ndcg"] for r in valid) / len(valid),
            "avg_latency": sum(r["total_latency"] for r in results) / len(results)
        }

    n_avg = calculate_averages(naive_results)
    a_avg = calculate_averages(adv_results)

    print("\n==================================================")
    print("📊 ADİL BENCHMARK DEĞERLENDİRME SONUÇLARI")
    print("==================================================")
    print(f"{'Metrik':<20} | {'Naive RAG Baseline':<20} | {'Advanced RAG':<20}")
    print("-" * 66)
    print(f"{'Precision@' + str(TOP_K):<20} | {n_avg.get('precision', 0):<20.4f} | {a_avg.get('precision', 0):<20.4f}")
    print(f"{'Recall@' + str(TOP_K):<20} | {n_avg.get('recall', 0):<20.4f} | {a_avg.get('recall', 0):<20.4f}")
    print(f"{'MRR':<20} | {n_avg.get('mrr', 0):<20.4f} | {a_avg.get('mrr', 0):<20.4f}")
    print(f"{'NDCG@' + str(TOP_K):<20} | {n_avg.get('ndcg', 0):<20.4f} | {a_avg.get('ndcg', 0):<20.4f}")
    print(f"{'Ort. Süre (sn)':<20} | {n_avg.get('avg_latency', 0):<20.4f} | {a_avg.get('avg_latency', 0):<20.4f}")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()