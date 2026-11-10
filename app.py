import os
import tempfile
import streamlit as st

# Orkestratör boru hattını içe aktarıyoruz
from advanced_rag import AdvancedRAGPipeline

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="DocPrecision-RAG Engine",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 DocPrecision-RAG Engine")
st.caption("Semantic Chunking + Hybrid Search (BM25 + BGE) + Groq Cloud API LLM")

# --- PIPELINE VE OTURUM DURUMU (SESSION STATE) İLKELENMESİ ---

# Önbellekleme mekanizmasını temiz şekilde yönetmek için ttl ekliyoruz veya gerektiğinde sıfırlıyoruz
@st.cache_resource(show_spinner="Model ve İndeksleme Altyapısı Yükleniyor...")
def load_rag_pipeline():
    return AdvancedRAGPipeline()

# Pipeline nesnesini TEK BİR DEFA başlatıyoruz
pipeline = load_rag_pipeline()

# Oturum geçmişi takibi
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_ingested" not in st.session_state:
    st.session_state.document_ingested = False

# --- SOL YAN PANEL (SIDEBAR): BELGE YÜKLEME ---
with st.sidebar:
    st.header("📄 Doküman Yönetimi")
    uploaded_file = st.file_uploader("Bir teknik PDF dokümanı yükleyin:", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Dokümanı İndeksle ve İşle", type="primary"):
            with st.spinner("📄 PDF okunuyor, anlamsal parçalanıyor ve indeksleniyor..."), tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                # Geçici dosyaya kaydet
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                tmp_file.close()

                # Boru hattına veriyi yükle
                try:
                    pipeline.ingest_document(tmp_path)
                    st.session_state.document_ingested = True
                    st.success(f"✅ Doküman başarıyla işlendi! ({len(pipeline.chunks)} anlamsal chunk üretildi)")
                except Exception as e:
                    st.error(f"❌ Doküman işlenirken hata oluştu: {str(e)}")
                finally:
                    # Geçici dosyayı temizle
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.divider()
    st.markdown("### ⚙️ Parametreler")
    top_k_retrieval = st.slider("Hybrid Search Aday Sayısı (top_k_retrieval)", min_value=3, max_value=10, value=6)
    top_k_rerank = st.slider("Reranker Seçim Sayısı (top_k_rerank)", min_value=1, max_value=5, value=2)

    # Önbelleği Zorla Sıfırlama Butonu
    st.divider()
    if st.button("🔄 Model Önbelleğini Sıfırla"):
        st.cache_resource.clear()
        st.success("Önbellek temizlendi! Sayfayı yenileyin.")

# --- ANA EKRAN: SOHBET ARAYÜZÜ ---

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "retrieved_chunks" in message:
            with st.expander("🔍 Kullanılan Bağlam Parçaları & Reranker Skorları"):
                for idx, chunk_info in enumerate(message["retrieved_chunks"]):
                    score = chunk_info.get("rerank_score", 0.0)
                    text = chunk_info["chunk"]["text"]
                    st.markdown(f"**Parça #{idx+1}** *(Alakalılık Skoru: `{score:.4f}`)*")
                    st.info(text)

# Kullanıcı yeni soru sorduğunda
if user_input := st.chat_input("Doküman hakkında sorunuzu yazın..."):

    if not st.session_state.document_ingested:
        st.warning("⚠️ Lütfen önce sol panelden bir PDF yükleyip 'Dokümanı İndeksle ve İşle' butonuna basın.")
    else:
        # 1. Kullanıcı mesajını göster ve hafızaya kaydet
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Yanıt üretme aşaması
        with st.chat_message("assistant"), st.spinner("🔍 Hibrit arama yapılıyor ve Groq API ile yanıt üretiliyor..."):
            res = pipeline.query(
                user_query=user_input,
                top_k_retrieval=top_k_retrieval,
                top_k_rerank=top_k_rerank
            )
            answer_text = res["answer"]
            retrieved_chunks = res["retrieved_chunks"]

            st.markdown(answer_text)

            # Seçilen bağlam parçalarını akordeonda göster
            with st.expander("🔍 Kullanılan Bağlam Parçaları & Reranker Skorları"):
                for idx, chunk_info in enumerate(retrieved_chunks):
                    score = chunk_info.get("rerank_score", 0.0)
                    text = chunk_info["chunk"]["text"]
                    st.markdown(f"**Parça #{idx+1}** *(Alakalılık Skoru: `{score:.4f}`)*")
                    st.info(text)

        # 3. Asistan yanıtını hafızaya kaydet
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "retrieved_chunks": retrieved_chunks
        })