import os 
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    print ("Modeller yapılandırılıyor...")
    
    #1. LLM yapılandırması
    # request_timeout süresini ilk yüklemeler için geniş tutuyoruz.
    Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
    
    #2. Embedding model yapılandırması
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    
    print ("Doküman okunuyor...")
    
    # data/ klasörğndeki PDF/metin dosyalarını oku
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Data klasoru bulunamadi, olusturuldu. Lutfen icine bir PDF ekleyin.")
        
        documents = SimpleDirectoryReader("data").load_data() #Verdiğim klasördeki tüm dokümanları bul, metinlerini ve sayfa bilgilerini ayıkla, hafızaya birer nesne olarak al
        if not documents:
            print("Data klasoru bos. Lutfen icine bir PDF ekleyin.")
            return
    
    print ("Dokuman sayfası yuklendi.")
    
    #3.Indexleme (Naive Chunking and Vector DB)
    
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
    
    print("Getirilen kaynak parçalar:")
    for source in response.source_nodes:
        print("\nKaynak Parça:\n", source.node.get_text())
        print("Kaynak Dosya: ", source.node.source_text)
        print("Sayfa Numarası: ", source.node.extra_info.get("page_number", "Bilinmiyor"))
    