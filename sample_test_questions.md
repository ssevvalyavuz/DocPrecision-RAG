# DocPrecision-RAG: X-200 Teknik Dokümanı Test Soruları ve Cevap Anahtarı

Bu doküman, **X-200 Endüstriyel Kazan Sistemi (V4.2)** teknik kılavuz PDF'i üzerinden RAG (Retrieval-Augmented Generation) sisteminizin doğruluğunu, arama yeteneğini ve hallüsinasyon (uydurma bilgi) korumasını test etmek amacıyla hazırlanmış örnek soru-cevap setlerini içerir.

---

## 📌 1. Doğrudan Bilgi Çıkarma ve Teknik Parametreler

Dokümandaki spesifik teknik değerleri, aralıkları ve malzeme bilgilerini doğrudan doğrulamak için kullanılan sorular.

### **Soru 1: X-200 modelinin gövde yapısı ve eşanjör kaplamasında hangi malzemeler kullanılmıştır?**
* **Cevap:** Kazan gövdesi 316L paslanmaz çelikten imal edilmiş olup, eşanjör plaka boruları titanyum kaplamalıdır.
* **Kaynak Bölüm:** *1. Genel Çalışma Prensipleri ve Yapısal Özellikler*

### **Soru 2: Kazan besleme suyu sıcaklığı ve standart çalışma basınç aralığı ne olmalıdır?**
* **Cevap:** Optimum çalışma koşullarında kazan besleme suyu sıcaklığı 85 °C seviyesinde tutulmalıdır. Standart çalışma basınç aralığı ise 4,2 bar ile 5,5 bar arasındadır.
* **Kaynak Bölüm:** *1. Genel Çalışma Prensipleri ve Yapısal Özellikler*

---

## 🚨 2. Güvenlik Protokolleri ve Acil Durum Yönetimi

Sıcaklık ve basınç limit aşımlarında sistemin ve operatörün alması gereken aksiyonları sorgulayan sorular.

### **Soru 3: Kazan iç sıcaklığı 115 °C'ye ulaştığında ve basınç 5.8 bar sınırına geldiğinde sistem nasıl tepki verir?**
* **Cevap:** 
  * 115 °C'de: Kontrol panosundaki sarı ikaz lambası sürekli yanar ve sesli uyarı devreye girer.
  * 5.8 bar'da: Solenoid V-10 tahliye valfi otomatik açılarak fazla basıncı atmosfere salar (Otomatik Tahliye).
* **Kaynak Bölüm:** *2. Güvenlik Protokolleri ve Kritik Limitler*

### **Soru 4: SCRAM (Acil Kapatma) hangi değerlerde tetiklenir ve bu durumda operatörün yapması gereken manuel müdahale nedir?**
* **Cevap:** Kritik eşik olan 135 °C sıcaklık veya 6.2 bar basınç aşıldığında SCRAM tetiklenir. Bu durumda operatörün V-12 ana valfini manuel olarak kapatması gerekir.
* **Kaynak Bölüm:** *2. Güvenlik Protokolleri ve Kritik Limitler*

---

## 🛠️ 3. Arıza Kodları ve Troubleshooting (Tablo Aramaları)

Arıza kodları tablosundan doğrudan müdahale adımlarını sorgulayan ve BM25/Dense arama başarımını ölçen sorular.

### **Soru 5: Panoda E-102 ve E-309 arıza kodları belirdiğinde yapılacak müdahaleler nelerdir?**
* **Cevap:** 
  * **E-102 (Düşük Su Seviyesi):** P-04 sigortası kontrol edilmeli ve emiş hattı vanası açılmalıdır.
  * **E-309 (Alev Duyarga Arızası):** Sensör lensi izopropil alkol ile temizlenmelidir.
* **Kaynak Bölüm:** *3. Arıza Kodları (Error Codes) ve Müdahale Adımları*

### **Soru 6: E-204 kritik sıcaklık hatası alındığında sistem kaç dakika soğumaya bırakılmalıdır?**
* **Cevap:** V-12 ana valfi kapatılmalı ve sistem 45 dakika soğumaya bırakılmalıdır.
* **Kaynak Bölüm:** *3. Arıza Kodları (Error Codes) ve Müdahale Adımları*

---

## 📅 4. Periyodik Bakım ve Otomasyon

Bakım periyotları, sorumluluk matrisi ve elektrik altyapısını sorgulayan sorular.

### **Soru 7: Kazan taş ve tortu temizliği (Descaling) hangi periyotta ve kimin sorumluluğunda yapılmalıdır?**
* **Cevap:** 6 Aylık periyotta, Kimyasal yıkama (Descaling) yöntemiyle Yetkili Servis tarafından yapılmalıdır.
* **Kaynak Bölüm:** *4. Periyodik Bakım ve Kontrol Çizelgesi*

### **Soru 8: Kontrol panosunun şebeke gerilimi, topraklama direnci limiti ve topraklama sistemi tipi nedir?**
* **Cevap:** Kontrol panosu 380V AC 50Hz trifaze şebeke gerilimi ile beslenir. Topraklama direnci maksimum 2 Ohm seviyesinde olmalı ve TN-S topraklama sistemi uygulanmalıdır (nötr ve toprak birleştirilmemelidir).
* **Kaynak Bölüm:** *5. Elektrik ve Otomasyon Bağlantı Kuralları*

---

## 🚫 5. Negatif / Hallüsinasyon Koruma Testi (Strict RAG Guardrails)

Dokümanda bulunmayan bilgileri sorgulayarak LLM'in uydurma yapıp yapmadığını doğrulayan sorular.

### **Soru 9: X-200 kazanının garanti süresi kaç yıldır ve yetkili servis çağrı merkezi numarası nedir?**
* **Cevap (Beklenen):** Verilen bağlamda garanti süresi veya yetkili servis telefon numarasına dair bir bilgi yer almamaktadır.
* **Amaç:** Modelin uydurma numara veya garanti süresi üretmesini engellemek.

### **Soru 10: Kazan brülöründe doğal gaz yerine motorin kullanmak için hangi dönüşüm kiti gereklidir?**
* **Cevap (Beklenen):** Dokümanda brülör yakıt dönüşüm kitine veya motorin kullanımına ilişkin bir bilgi bulunmamaktadır.
* **Amaç:** Doküman dışı teknik varsayımları sınırlamak.

---

## 🧪 Test Komutları ve Kullanım Önerisi

Bu soruları Streamlit sohbet arayüzünde tek tek sorarak şu 3 kriteri doğrulayabilirsiniz:
1. **Context Retrieval:** Cevap altındaki akordeon menüsünde doğru tablonun/parçanın (chunk) getirilip getirilmediği (Alakalılık skoru).
2. **Answer Precision:** LLM'in cevabı 2-3 cümlede net ve bağlama sadık verip vermediği.
3. **Hallucination Rate:** 9 ve 10. sorularda modelin dürüstçe *"Bağlamda bulunamadı"* yanıtı verip vermediği.
```

eof

I have generated the `README.md` file containing categorized sample test questions and explicit reference answers based on the PDF contents.

You can copy these test questions directly into your Streamlit interface to evaluate your pipeline's retrieval and answer accuracy.