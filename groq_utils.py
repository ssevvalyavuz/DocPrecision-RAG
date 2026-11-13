import re
import time
import random
from typing import List, Dict, Any, Optional
from groq import Groq, RateLimitError

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = RateLimitError


class RateLimitWaitExceededError(RuntimeError):
    """429 bekleme süresi belirlenen maksimum sınırı (60 sn) aştığında fırlatılır."""
    pass


def _parse_retry_after_header(error: Exception) -> Optional[float]:
    """HTTP response header içindeki Retry-After değerini saniye cinsinden döndürür."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    header_val = headers.get("retry-after") or headers.get("Retry-After")
    if header_val is None:
        return None

    try:
        parsed = float(header_val)
        return parsed if parsed > 0 else None
    except ValueError:
        return None


def _parse_wait_time_from_message(error_message: str) -> Optional[float]:
    """Hata mesajı içindeki 'try again in X.XXXXs' (veya ms) süresini saniye cinsinden döndürür."""
    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s)\b", error_message, re.IGNORECASE)
    if not match:
        return None

    value = float(match.group(1))
    seconds = (value / 1000.0) if match.group(2).lower() == "ms" else value
    return seconds if seconds > 0 else None


def _extract_wait_time_from_error(error: Exception) -> Optional[float]:
    """
    429 hatasından bekleme süresini (saniye cinsinden) sırasıyla:
    1. HTTP 'Retry-After' header'ından
    2. Hata mesajı içindeki 'Please try again in X.XXXXs' (veya ms) metninden
    ayrıştırır. Bulunamazsa None döner.
    """
    header_wait = _parse_retry_after_header(error)
    if header_wait is not None:
        return header_wait

    return _parse_wait_time_from_message(str(error))


def _sleep_with_backoff(
    error: Exception,
    attempt: int,
    max_retries: int,
    base_delay: float,
    safety_buffer: float,
    max_wait_seconds: float = 60.0
) -> None:
    """Bekleme süresini hesaplar; 60 sn sınırını aşıyorsa işlemi kontrollü durdurur."""
    extracted_wait = _extract_wait_time_from_error(error)
    if extracted_wait is not None:
        wait_time = extracted_wait + safety_buffer
    else:
        wait_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 0.6)

    if wait_time > max_wait_seconds:
        raise RateLimitWaitExceededError(
            f"🛑 Groq 429 bekleme süresi ({wait_time:.2f} sn) izin verilen {max_wait_seconds:.0f} sn sınırını aştı. "
            f"İşlem kontrollü olarak durduruldu."
        ) from error

    print(
        f"⏳ Groq Rate Limit (429) aşıldı. {wait_time:.2f} sn bekleniyor... "
        f"(Deneme {attempt + 1}/{max_retries})"
    )
    time.sleep(wait_time)


def call_groq_with_retry(
    client: Groq,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
    safety_buffer: float = 0.25,
    max_wait_seconds: float = 60.0
) -> Any:
    """
    Groq chat completion çağrısını yalnızca HTTP 429 (RateLimitError) için
    otomatik tekrar deneme (retry) korumasıyla çalıştırır.
    Diğer tüm API/kimlik/model hatalarını yakalamadan doğrudan yükseltir.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                messages=messages,
                model=model,
            )
        except RateLimitError as exc:
            if attempt == max_retries:
                raise
            _sleep_with_backoff(exc, attempt, max_retries, base_delay, safety_buffer, max_wait_seconds)


def call_llamaindex_llm_with_retry(
    llm: Any,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
    safety_buffer: float = 0.25,
    max_wait_seconds: float = 60.0
) -> Any:
    """advanced_rag.py içindeki OpenAILike (llm.complete) çağrısı için 429 retry koruması."""
    for attempt in range(max_retries + 1):
        try:
            return llm.complete(prompt)
        except (OpenAIRateLimitError, RateLimitError) as exc:
            if attempt == max_retries:
                raise
            _sleep_with_backoff(exc, attempt, max_retries, base_delay, safety_buffer, max_wait_seconds)