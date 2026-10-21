import os 
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # HuggingFace Hub token kullanımını devre dışı bırakır.
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""  # Benden token arama, bu halka açık modeli anonim olarak doğrudan indir.
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # HuggingFace tokenizer paralel çalışmasını devre dışı bırakır.


def main():
    print ("Modeller yapılandırılıyor...")
    
    # LLM yapılandırması
    # request_timeout süresini ilk yüklemeler için geniş tutuyoruz.
    Settings.llm = Ollama(model="llama3.2", request_timeout=300.0, temperature=0.1 )
    
    #. Embedding model yapılandırması
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", embed_batch_size=10)
    
    # Akıllı parçalayıcı (SentenceSplitter) Tanımlama
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # 512 token boyutunda parçalar, 50 token örtüşme ile
    Settings.text_splitter = text_splitter
    
    
#  Dokümanları Yükleme
    print("Doküman okunuyor...")
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        print("❌ 'data' klasörü bulunamadı ve yeni oluşturuldu. Lütfen içine bir PDF koyun.")
        return

    pdf_path = "data/manual.pdf"  # Örnek PDF dosya yolu
    if not os.path.isfile(pdf_path):
        print(f"'{pdf_path}' dosyası bulunamadı. Lütfen 'data' klasörüne bir PDF ekleyin.")
        return
    
    loader = PyMuPDFReader()
    documents = loader.load_data(pdf_path)
    
# Parçalama (Chunking) İşlemini Manuel Olarak da İnceleyelim
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f" Doküman toplam {len(documents)} sayfadan {len(nodes)} adet mantıksal parçaya (Node) ayrıldı.")    

    if not documents:
        print("'data' klasöründe okunabilir doküman bulunamadı! Lütfen bir PDF ekleyin.")
        return

    print(f" {len(documents)} doküman sayfası yüklendi.")    
    #Indexleme (Naive Chunking and Vector DB)
    # İlk test için CPU'yu yormamak adına ilk 3 sayfa
    test_docs = documents[:3] if len(documents) > 3 else documents
    print(f"Toplam {len(documents)} sayfadan test için ilk {len(test_docs)} sayfa seçildi.")
    print("Indexleme başlatılıyor...")
    
    # Ham metin sayfalarını varsayılan token boyutuna (chunk) böler, 
    # BGE embedding modeliyle vektörlere (sayısal koordinatlara) dönüştürür 
    # ve arama yapılabilir bir vektör veritabanı (Vector DB) fihristi oluşturur.
    index = VectorStoreIndex.from_documents(documents)
    
    # Kullanıcı sorusunu vektöre çeviren, en benzer 2 metin parçasını (top-k=2) 
    # depodan çekip Ollama LLM'e bağlam (context) olarak ileten sorgu boru hattını kurar.
    query_engine = index.as_query_engine(similarity_top_k=2)    
    print("\n" + "="*50)
    print("Soru sorma motoru hazır. Sorularınızı sorabilirsiniz. Çıkmak için 'exit' yazın.")    
    
    while True:
        user_input = input("\nSoru: ")
        if user_input.lower() == "exit":
            print("Çıkış yapılıyor...")
            break
        
        response = query_engine.query(user_input) #Soru sorma motoruna soruyu gönderir ve yanıt alır.
        print("\nYanıt: ", response) #Yanıtı ekrana yazdırır.    
        
        # --- RESPONSE NESNESİ DETAYLI DENETİMİ ---
        print("\n" + "=" * 50)
        print("🔍 RESPONSE İÇERİK ANALİZİ")
        print("=" * 50)
        
        # 1. Genel metadata (varsa model ve sistem bilgileri)
        print(f"Genel Metadata: {response.metadata}")

        # 2. LLM'e beslenen kaynak parçalar (source_nodes)
        print(f"\nKullanılan Kaynak Parça Sayısı: {len(response.source_nodes)}")
        
        for idx, source in enumerate(response.source_nodes, 1):
            print(f"\n--- [Parça {idx}] ---")
            print(f"• Benzerlik Skoru : {source.score:.4f}")
            print(f"• Node ID         : {source.node.node_id}")
            print(f"• Dosya Adı       : {source.node.metadata.get('file_name', 'Bilinmiyor')}")
            print(f"• Sayfa Numarası  : {source.node.metadata.get('page_label', source.node.metadata.get('page_number', 'Bilinmiyor'))}")
            print("• Metin İçeriği   :")
            print(f"  \"{source.node.get_text().strip()[:200]}...\"")
            
        print("=" * 50 + "\n")
    
if __name__ == "__main__":
    main()