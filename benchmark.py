import os
import sys
import ast
import json
import time
import types
import asyncio
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

# Ragas 0.4.3 içindeki kullanılmayan VertexAI import hatasını engeller
_mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
_mock_vertex.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = _mock_vertex

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall

from naive_rag import NaiveRAGPipeline
from advanced_rag import AdvancedRAGPipeline

DATASET_PATH = os.path.join("evaluation", "questions.json")
CHECKPOINT_PATH = os.path.join("evaluation", "ragas_collections_checkpoint.json")
OUTPUT_PATH = os.path.join("evaluation", "ragas_collections_results.json")
PDF_PATH = os.path.join("data", "X200_Kapsamli_Teknik_Kilavuz.pdf")
TOP_K = 3

# Önce tek bir soru üzerinde doğrulama yapmak için True; başarılı olursa False yapın
TEST_SINGLE_QUESTION = False 


# --- 1. RAGAS 0.4.3 EVALUATOR LLM VE COLLECTIONS METRİKLERİ ---

def _create_ragas_metrics() -> Tuple[ContextPrecision, ContextRecall]:
    """Groq OpenAI-uyumlu istemci üzerinden Ragas 0.4.3 Collections metriklerini başlatır."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(".env dosyasında GROQ_API_KEY bulunamadı!")

    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    async_client = AsyncOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        max_retries=3
    )

    evaluator_llm = llm_factory(groq_model, client=async_client)
    precision_metric = ContextPrecision(llm=evaluator_llm)
    recall_metric = ContextRecall(llm=evaluator_llm)

    return precision_metric, recall_metric


async def _score_with_ragas_async(
    precision_metric: ContextPrecision,
    recall_metric: ContextRecall,
    user_input: str,
    reference: str,
    retrieved_contexts: List[str]
) -> Dict[str, Any]:
    """Ragas 0.4.3 Collections API (ascore) ile ContextPrecision ve ContextRecall hesaplar."""
    precision_res = await precision_metric.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )
    recall_res = await recall_metric.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )

    p_val = float(precision_res.value) if getattr(precision_res, "value", None) is not None else None
    r_val = float(recall_res.value) if getattr(recall_res, "value", None) is not None else None

    return {
        "context_precision": p_val,
        "context_recall": r_val
    }


def evaluate_retrieval_ragas(
    precision_metric: ContextPrecision,
    recall_metric: ContextRecall,
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
    k: int
) -> Dict[str, Any]:
    """Top-K bağlam parçalarını Ragas Collections API ile senkron olarak değerlendirir."""
    k_contexts = retrieved_contexts[:k]
    return asyncio.run(
        _score_with_ragas_async(
            precision_metric=precision_metric,
            recall_metric=recall_metric,
            user_input=user_input,
            reference=reference,
            retrieved_contexts=k_contexts
        )
    )


# --- 2. JSON KAYIT FORMATLAYICILAR VE METİN ÇIKARICILAR ---

def _extract_advanced_clean_text(item: Any) -> str:
    """Advanced RAG chunk nesnesinden saf metni (string dict içinden ayrıştırarak) çıkarır."""
    if not isinstance(item, dict):
        return str(item)

    chunk_wrapper = item.get("chunk")
    raw_text_str = chunk_wrapper.get("text", "") if isinstance(chunk_wrapper, dict) else str(chunk_wrapper or "")

    if isinstance(raw_text_str, str) and raw_text_str.strip().startswith("{"):
        try:
            candidate = ast.literal_eval(raw_text_str.strip())
            if isinstance(candidate, dict):
                inner_chunk = candidate.get("chunk") if isinstance(candidate.get("chunk"), dict) else {}
                return str(inner_chunk.get("text", candidate.get("text", "")))
        except (ValueError, SyntaxError, TypeError):
            pass

    return raw_text_str


def _format_naive_chunks(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for item in raw_chunks:
        if isinstance(item, dict):
            formatted.append({
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text", ""),
                "similarity_score": round(float(item["similarity_score"]), 4) if item.get("similarity_score") is not None else None
            })
        else:
            formatted.append({
                "chunk_id": None,
                "text": str(item),
                "similarity_score": None
            })
    return formatted


def _format_advanced_chunks(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            formatted.append({
                "chunk_id": None,
                "text": str(item),
                "rrf_score": None,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
                "rerank_score": None
            })
            continue

        raw_rerank = item.get("rerank_score")
        rerank_score = round(float(raw_rerank), 4) if raw_rerank is not None else None

        chunk_wrapper = item.get("chunk")
        raw_text_str = chunk_wrapper.get("text", "") if isinstance(chunk_wrapper, dict) else str(chunk_wrapper or "")

        parsed_outer = None
        if isinstance(raw_text_str, str) and raw_text_str.strip().startswith("{"):
            try:
                candidate = ast.literal_eval(raw_text_str.strip())
                if isinstance(candidate, dict):
                    parsed_outer = candidate
            except (ValueError, SyntaxError, TypeError):
                parsed_outer = None

        if parsed_outer is not None:
            inner_chunk = parsed_outer.get("chunk") if isinstance(parsed_outer.get("chunk"), dict) else {}
            chunk_id = inner_chunk.get("chunk_id", parsed_outer.get("chunk_id"))
            clean_text = inner_chunk.get("text", parsed_outer.get("text", ""))

            rrf_val = parsed_outer.get("rrf_score")
            dense_s_val = parsed_outer.get("dense_score")
            sparse_s_val = parsed_outer.get("sparse_score")

            formatted.append({
                "chunk_id": chunk_id,
                "text": clean_text,
                "rrf_score": round(float(rrf_val), 4) if rrf_val is not None else None,
                "dense_rank": parsed_outer.get("dense_rank"),
                "sparse_rank": parsed_outer.get("sparse_rank"),
                "dense_score": round(float(dense_s_val), 4) if dense_s_val is not None else None,
                "sparse_score": round(float(sparse_s_val), 4) if sparse_s_val is not None else None,
                "rerank_score": rerank_score
            })
        else:
            fallback_id = chunk_wrapper.get("chunk_id") if isinstance(chunk_wrapper, dict) else item.get("chunk_id")
            formatted.append({
                "chunk_id": fallback_id,
                "text": raw_text_str,
                "rrf_score": None,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
                "rerank_score": rerank_score
            })

    return formatted


# --- 3. CHECKPOINT VE ÖZET HESAPLAMA ---

def _is_record_complete(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict) or "id" not in record:
        return False

    naive_block = record.get("naive_rag")
    adv_block = record.get("advanced_rag")
    if not isinstance(naive_block, dict) or not isinstance(adv_block, dict):
        return False

    required_keys = {"retrieved_chunks", "retrieval_latency", "context_precision", "context_recall"}
    has_naive = required_keys.issubset(naive_block.keys()) and naive_block.get("context_precision") is not None
    has_adv = required_keys.issubset(adv_block.keys()) and adv_block.get("context_precision") is not None

    return has_naive and has_adv


def _load_checkpoint(checkpoint_path: str) -> Dict[Any, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        return {}

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("results", []) if isinstance(data, dict) else data
        if not isinstance(records, list):
            return {}
        return {rec["id"]: rec for rec in records if _is_record_complete(rec)}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_checkpoint(
    checkpoint_path: str,
    pdf_path: str,
    top_k: int,
    total_questions: int,
    records: List[Dict[str, Any]]
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "metadata": {
            "evaluation_framework": "ragas_0.4.3_collections",
            "evaluation_mode": "retrieval_only",
            "pdf_path": pdf_path,
            "top_k": top_k,
            "total_questions": total_questions,
            "completed_questions": len(records)
        },
        "results": records
    }
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        json.dump(payload, out_f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)


def _calculate_summary_from_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    def _avg_for_system(sys_key: str) -> Dict[str, Any]:
        valid = []
        latencies = []
        for rec in records:
            sys_data = rec.get(sys_key, {})
            if sys_data.get("retrieval_latency") is not None:
                latencies.append(sys_data["retrieval_latency"])
            if sys_data.get("context_precision") is not None and sys_data.get("context_recall") is not None:
                valid.append(sys_data)

        avg_latency = (sum(latencies) / len(latencies)) if latencies else None

        if not valid:
            return {
                "context_precision": None,
                "context_recall": None,
                "avg_retrieval_latency": avg_latency
            }

        return {
            "context_precision": sum(r["context_precision"] for r in valid) / len(valid),
            "context_recall": sum(r["context_recall"] for r in valid) / len(valid),
            "avg_retrieval_latency": avg_latency
        }

    return {
        "naive_rag": _avg_for_system("naive_rag"),
        "advanced_rag": _avg_for_system("advanced_rag")
    }


# --- 4. SAF RETRIEVAL VE BENCHMARK YARDIMCI FONKSİYONLARI ---

def _initialize_pipelines(pdf_path: str) -> Tuple[NaiveRAGPipeline, AdvancedRAGPipeline]:
    print("--- [1/2] Naive RAG İndeksleniyor ---")
    naive_pipe = NaiveRAGPipeline()
    naive_pipe.ingest_document(pdf_path)

    print("\n--- [2/2] Advanced RAG İndeksleniyor ---")
    adv_pipe = AdvancedRAGPipeline()
    adv_pipe.ingest_document(pdf_path)

    return naive_pipe, adv_pipe


def _retrieve_naive_safe(
    naive_pipe: NaiveRAGPipeline,
    query_text: str,
    top_k: int
) -> Tuple[List[Dict[str, Any]], float]:
    """naive_rag.py dosyasına dokunmadan Groq API çağırmadan Naive retrieval yapar."""
    if hasattr(naive_pipe, "retrieve_only"):
        t0 = time.perf_counter()
        res = naive_pipe.retrieve_only(query_text, top_k=top_k)
        latency = res.get("retrieval_latency", round(time.perf_counter() - t0, 4))
        return res.get("retrieved_chunks", []), float(latency)

    if not naive_pipe.chunks or not naive_pipe.chunk_embeddings:
        raise ValueError("Lütfen önce bir doküman indeksleyin!")

    t0 = time.perf_counter()
    query_embedding = naive_pipe.embed_model.get_query_embedding(query_text)
    similarities = naive_pipe._cosine_similarity(query_embedding, naive_pipe.chunk_embeddings)
    top_indices = similarities.argsort()[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        chunk_data = naive_pipe.chunks[idx]
        retrieved.append({
            "chunk_id": chunk_data["chunk_id"],
            "text": chunk_data["text"],
            "similarity_score": float(similarities[idx])
        })
    return retrieved, round(time.perf_counter() - t0, 4)


def _retrieve_advanced_safe(
    adv_pipe: AdvancedRAGPipeline,
    query_text: str,
    top_k_retrieval: int,
    top_k_rerank: int
) -> Tuple[List[Dict[str, Any]], float]:
    """advanced_rag.py dosyasına dokunmadan Groq API çağırmadan Advanced retrieval yapar."""
    if hasattr(adv_pipe, "retrieve_only"):
        t0 = time.perf_counter()
        res = adv_pipe.retrieve_only(query_text, top_k_retrieval=top_k_retrieval, top_k_rerank=top_k_rerank)
        latency = res.get("retrieval_latency", round(time.perf_counter() - t0, 4))
        return res.get("retrieved_chunks", []), float(latency)

    if not adv_pipe.retriever:
        raise ValueError("Lütfen önce doküman yükleyin")

    t0 = time.perf_counter()
    candidate_results = adv_pipe.retriever.retrieve(query_text, top_k=top_k_retrieval)
    selected_results = candidate_results[:top_k_rerank]

    retrieved = []
    for res in selected_results:
        chunk_text = res.node.get_content() if hasattr(res, "node") else str(res)
        score = getattr(res, "score", 0.0)
        retrieved.append({
            "chunk": {"text": chunk_text},
            "rerank_score": score
        })
    return retrieved, round(time.perf_counter() - t0, 4)


def _build_system_result_entry(
    formatted_chunks: List[Dict[str, Any]],
    latency: float,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "retrieved_chunks": formatted_chunks,
        "retrieval_latency": round(float(latency), 4),
        "context_precision": round(metrics["context_precision"], 4) if metrics["context_precision"] is not None else None,
        "context_recall": round(metrics["context_recall"], 4) if metrics["context_recall"] is not None else None
    }


def _evaluate_single_question(
    q: Dict[str, Any],
    naive_pipe: NaiveRAGPipeline,
    adv_pipe: AdvancedRAGPipeline,
    precision_metric: ContextPrecision,
    recall_metric: ContextRecall,
    top_k: int
) -> Dict[str, Any]:
    q_id = q["id"]
    q_text = q["question"]
    reference = q.get("ground_truth", "")

    print(f"❓ [Soru #{q_id}] {q_text}")

    # 1. Naive RAG Retrieval-Only (RAG generate() çalışmaz)
    n_raw_chunks, n_latency = _retrieve_naive_safe(naive_pipe, q_text, top_k=top_k)
    n_contexts = [c["text"] if isinstance(c, dict) else str(c) for c in n_raw_chunks]
    n_metrics = evaluate_retrieval_ragas(
        precision_metric, recall_metric, q_text, reference, n_contexts, k=top_k
    )

    # 2. Advanced RAG Retrieval-Only (RAG generate() çalışmaz)
    a_raw_chunks, a_latency = _retrieve_advanced_safe(
        adv_pipe, q_text, top_k_retrieval=6, top_k_rerank=top_k
    )
    a_contexts = [_extract_advanced_clean_text(c) for c in a_raw_chunks]
    a_metrics = evaluate_retrieval_ragas(
        precision_metric, recall_metric, q_text, reference, a_contexts, k=top_k
    )

    print(
        f"   📊 Naive    -> Precision: {n_metrics['context_precision']:.4f} | Recall: {n_metrics['context_recall']:.4f}\n"
        f"   📊 Advanced -> Precision: {a_metrics['context_precision']:.4f} | Recall: {a_metrics['context_recall']:.4f}"
    )

    return {
        "id": q_id,
        "category": q.get("category", "unknown"),
        "user_input": q_text,
        "reference": reference,
        "naive_rag": _build_system_result_entry(_format_naive_chunks(n_raw_chunks), n_latency, n_metrics),
        "advanced_rag": _build_system_result_entry(_format_advanced_chunks(a_raw_chunks), a_latency, a_metrics)
    }


def _run_evaluation_loop(
    target_questions: List[Dict[str, Any]],
    all_questions: List[Dict[str, Any]],
    completed_map: Dict[Any, Dict[str, Any]],
    naive_pipe: NaiveRAGPipeline,
    adv_pipe: AdvancedRAGPipeline,
    precision_metric: ContextPrecision,
    recall_metric: ContextRecall
) -> Tuple[List[Dict[str, Any]], bool]:
    print("\n⚡ RAGAS 0.4.3 COLLECTIONS RETRIEVAL EVALUATION ÇALIŞIYOR...\n")
    detailed_records = [completed_map[q["id"]] for q in all_questions if q["id"] in completed_map]

    try:
        for q in target_questions:
            q_id = q["id"]
            if q_id in completed_map:
                print(f"⏭️ [Soru #{q_id}] Checkpoint'te mevcut, atlanıyor...")
                continue

            record = _evaluate_single_question(
                q, naive_pipe, adv_pipe, precision_metric, recall_metric, TOP_K
            )
            completed_map[q_id] = record
            detailed_records = [completed_map[item["id"]] for item in all_questions if item["id"] in completed_map]
            _save_checkpoint(CHECKPOINT_PATH, PDF_PATH, TOP_K, len(all_questions), detailed_records)
            print(f"💾 [Soru #{q_id}] Checkpoint kaydedildi ({len(detailed_records)}/{len(all_questions)}).")

        return detailed_records, True
    except Exception as exc:
        print(f"\n⚠️ Ragas Collections değerlendirmesi hata nedeniyle durduruldu: {exc}")
        print(f"💾 Tamamlanan {len(detailed_records)} soru '{CHECKPOINT_PATH}' dosyasında güvenle korundu.")
        return detailed_records, False


def _print_retrieval_summary(summary: Dict[str, Dict[str, Any]], top_k: int) -> None:
    n_avg = summary.get("naive_rag", {})
    a_avg = summary.get("advanced_rag", {})

    def _fmt(val: Any) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    print("\n==========================================================")
    print(f"📊 RAGAS 0.4.3 (COLLECTIONS) RETRIEVAL SONUÇLARI (Top-{top_k})")
    print("==========================================================")
    print(f"{'Metrik':<26} | {'Naive RAG Baseline':<20} | {'Advanced RAG':<20}")
    print("-" * 72)
    print(f"{'Context Precision':<26} | {_fmt(n_avg.get('context_precision')):<20} | {_fmt(a_avg.get('context_precision')):<20}")
    print(f"{'Context Recall':<26} | {_fmt(n_avg.get('context_recall')):<20} | {_fmt(a_avg.get('context_recall')):<20}")
    print(f"{'Ort. Retrieval Süre (sn)':<26} | {_fmt(n_avg.get('avg_retrieval_latency')):<20} | {_fmt(a_avg.get('avg_retrieval_latency')):<20}")
    print("==========================================================")


def _save_final_results(
    output_path: str,
    pdf_path: str,
    top_k: int,
    total_questions: int,
    summary: Dict[str, Dict[str, Any]],
    records: List[Dict[str, Any]]
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_payload = {
        "metadata": {
            "evaluation_framework": "ragas_0.4.3_collections",
            "metrics": ["ContextPrecision", "ContextRecall"],
            "pdf_path": pdf_path,
            "top_k": top_k,
            "total_questions": total_questions,
            "evaluated_questions": len(records)
        },
        "summary": {
            "naive_rag": {k: (round(v, 4) if v is not None else None) for k, v in summary.get("naive_rag", {}).items()},
            "advanced_rag": {k: (round(v, 4) if v is not None else None) for k, v in summary.get("advanced_rag", {}).items()}
        },
        "results": records
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(export_payload, out_f, ensure_ascii=False, indent=2)

    print(f"\n💾 Ragas Collections sonuçları kaydedildi: {output_path}")
    print(f"📌 Ragas Collections checkpoint korundu: {CHECKPOINT_PATH}")


# --- 5. ANA ÇALIŞTIRICI ---

def run_benchmark() -> None:
    if not os.path.exists(DATASET_PATH) or not os.path.exists(PDF_PATH):
        raise FileNotFoundError("Gerekli veri veya PDF dosyası bulunamadı!")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    target_questions = all_questions[:1] if TEST_SINGLE_QUESTION else all_questions

    completed_map = _load_checkpoint(CHECKPOINT_PATH)
    detailed_records = [completed_map[q["id"]] for q in all_questions if q["id"] in completed_map]

    print("==================================================")
    print("🚀 DocPrecision-RAG | Ragas 0.4.3 Collections Benchmark")
    print(f"📄 PDF: {PDF_PATH}")
    print(f"🧪 Mod: {'TEK SORU DOĞRULAMA' if TEST_SINGLE_QUESTION else 'TÜM VERİ SETİ'}")
    print(f"❓ Hedef Soru: {len(target_questions)} / {len(all_questions)} | ✅ Checkpoint: {len(detailed_records)}")
    print(f"🎯 Top-K: {TOP_K} | 🚫 RAG generate(): KAPALI")
    print("==================================================\n")

    pending = [q for q in target_questions if q["id"] not in completed_map]
    if pending:
        precision_metric, recall_metric = _create_ragas_metrics()
        naive_pipe, adv_pipe = _initialize_pipelines(PDF_PATH)
        detailed_records, completed = _run_evaluation_loop(
            target_questions, all_questions, completed_map,
            naive_pipe, adv_pipe, precision_metric, recall_metric
        )
        if not completed:
            return
    else:
        print("✅ Hedef sorular zaten checkpoint dosyasında tamamlanmış görünüyor.\n")

    summary = _calculate_summary_from_records(detailed_records)
    _print_retrieval_summary(summary, TOP_K)
    _save_final_results(OUTPUT_PATH, PDF_PATH, TOP_K, len(all_questions), summary, detailed_records)


if __name__ == "__main__":
    run_benchmark()