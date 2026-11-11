import os
import urllib.request
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def setup_turkish_fonts():
    """
    DejaVuSans fontlarını proje içindeki fonts klasörüne indirip kaydeder.
    """
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    os.makedirs(font_dir, exist_ok=True)

    font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    font_bold_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    font_urls = {
        font_path: "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        font_bold_path: "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    }

    for path, url in font_urls.items():
        if not os.path.isfile(path):
            print(f"Türkçe destekli font indiriliyor: {os.path.basename(path)}")
            urllib.request.urlretrieve(url, path)

    pdfmetrics.registerFont(TTFont('TRFont', font_path))
    pdfmetrics.registerFont(TTFont('TRFont-Bold', font_bold_path))
    print("Font yüklemesi tamamlandı.")

def build_pdf(filename="X200_Kapsamli_Teknik_Kilavuz.pdf"):
    setup_turkish_fonts()
    
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Türkçe font entegrasyonlu özel stiller
    title_style = ParagraphStyle(
        'TRTitle',
        parent=styles['Heading1'],
        fontName='TRFont-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Center
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'TRH1',
        fontName='TRFont-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'TRBody',
        fontName='TRFont',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )
    
    table_header_style = ParagraphStyle(
        'TRTableHeader',
        fontName='TRFont-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TRTableCell',
        fontName='TRFont',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # --- BAŞLIK VE KÜNYE ---
    story.extend([
        Paragraph("ENDÜSTRİYEL KAZAN SİSTEMLERİ X-200", title_style),
        Paragraph("KAPSAMLI TEKNİK İŞLETİM VE BAKIM KILAVUZU (V4.2)", ParagraphStyle('Sub', parent=title_style, fontSize=11, textColor=colors.HexColor("#718096"), spaceAfter=10)),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=15),
    ])

    # --- BÖLÜM 1: GENEL TANIM VE ÇALIŞMA PRENSİPLERİ ---
    story.extend([
        Paragraph("1. Genel Çalışma Prensipleri ve Yapısal Özellikler", h1_style),
        Paragraph(
            "X-200 modeli, yüksek basınçlı endüstriyel buhar üretimi için özel olarak tasarlanmış modüler bir kazan sistemidir. "
            "Sistem, yüksek sıcaklık dayanımına sahip 316L paslanmaz çelik gövde yapısı ve titanyum kaplamalı eşanjör plaka boruları ile donatılmıştır. "
            "Standart çalışma basınç aralığı <b>4,2 bar ile 5,5 bar</b> arasındadır. Bu aralığın dışındaki işletim şartları, güvenlik mimarisi tarafından otomatik olarak engellenir.",
            body_style
        ),
        Paragraph(
            "Kazan içi su seviyesi sürekli olarak ultrasonik debimetreler ve elektrotlu seviye sensörleri ile izlenmektedir. "
            "Optimum çalışma koşullarında kazan besleme suyu sıcaklığı 85 °C seviyesinde tutulmalıdır.",
            body_style
        ),
    ])

    # --- BÖLÜM 2: GÜVENLİK PROTOKOLLERİ VE LİMİTLER ---
    story.extend([
        Paragraph("2. Güvenlik Protokolleri ve Kritik Limitler", h1_style),
        Paragraph(
            "Sistemin güvenli işletimi için 3 kademeli koruma protokolü tanımlanmıştır:",
            body_style
        ),
    ])
    
    guvenlik_maddeleri = [
        "<b>Kademeli Uyarı (115 °C):</b> Kazan iç sıcaklığı 115 °C değerine ulaştığında kontrol panosundaki sarı ikaz lambası sürekli yanar ve sesli uyarı devreye girer.",
        "<b>Otomatik Tahliye (125 °C / 5.8 bar):</b> Basınç 5,8 bar sınırına ulaştığında Solenoid V-10 tahliye valfi otomatik açılarak fazla basıncı atmosfere salar.",
        "<b>Acil Kapatma - SCRAM (135 °C / 6.2 bar):</b> Kritik eşik olan 135 °C sıcaklık veya 6.2 bar basınç aşıldığında acil kapatma sistemi (SCRAM) tetiklenir. Bu aşamada operatör V-12 ana valfini manuel olarak kapatmalıdır."
    ]
    for madde in guvenlik_maddeleri:
        story.append(Paragraph(f"• {madde}", body_style))

    story.append(Spacer(1, 10))

    # --- BÖLÜM 3: ARIZA KODLARI VE ÇÖZÜM TABLOSU ---
    story.append(Paragraph("3. Arıza Kodları (Error Codes) ve Müdahale Adımları", h1_style))
    
    ariza_data = [
        [Paragraph("Arıza Kodu", table_header_style), Paragraph("Arıza Tanımı", table_header_style), Paragraph("Olası Neden", table_header_style), Paragraph("Çözüm / Müdahale", table_header_style)],
        [Paragraph("E-101", table_cell_style), Paragraph("Aşırı Basınç Yükselmesi", table_cell_style), Paragraph("V-10 Valfi tıkalı veya duyarga arızalı", table_cell_style), Paragraph("V-10 valfini manuel açın, duyarga kalibrasyonunu kontrol edin.", table_cell_style)],
        [Paragraph("E-102", table_cell_style), Paragraph("Düşük Su Seviyesi", table_cell_style), Paragraph("Besleme pompası P-04 devre dışı", table_cell_style), Paragraph("P-04 sigortasını kontrol edin, emiş hattı vanasını açın.", table_cell_style)],
        [Paragraph("E-204", table_cell_style), Paragraph("Kritik Sıcaklık (SCRAM)", table_cell_style), Paragraph("Aşırı yük veya brülör kontrolden çıktı", table_cell_style), Paragraph("V-12 ana valfini kapatın, sistemi 45 dk soğumaya bırakın.", table_cell_style)],
        [Paragraph("E-309", table_cell_style), Paragraph("Alev Duyarga Arızası", table_cell_style), Paragraph("Fotosel sensör yüzeyi kirlenmiş", table_cell_style), Paragraph("Sensör lensini izopropil alkol ile temizleyin.", table_cell_style)],
        [Paragraph("E-412", table_cell_style), Paragraph("İletişim Kopukluğu (PLC)", table_cell_style), Paragraph("RS485 Modbus kablosu gevşek", table_cell_style), Paragraph("Pano klemens bağlantılarını ve sonlandırma direncini sıkın.", table_cell_style)]
    ]
    
    ariza_table = Table(ariza_data, colWidths=[65, 120, 150, 195])
    ariza_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C5282")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ariza_table)

    story.append(PageBreak()) # Sayfa Sonu

    # --- BÖLÜM 4: PERİYODİK BAKIM ÇİZELGESİ ---
    story.extend([
        Paragraph("4. Periyodik Bakım ve Kontrol Çizelgesi", h1_style),
        Paragraph(
            "Kazan sisteminin kullanım ömrünü uzatmak ve garanti kapsamını korumak için aşağıdaki bakım takvimine uyulması zorunludur.",
            body_style
        ),
    ])
    
    bakim_data = [
        [Paragraph("Periyot", table_header_style), Paragraph("Kontrol Noktası", table_header_style), Paragraph("Yapılacak İşlem", table_header_style), Paragraph("Sorumlu", table_header_style)],
        [Paragraph("Günlük", table_cell_style), Paragraph("Su Seviyesi ve Basınç", table_cell_style), Paragraph("Gösterge değerlerinin nominal aralıkta olduğu teyit edilir.", table_cell_style), Paragraph("Vardiya Operatörü", table_cell_style)],
        [Paragraph("Haftalık", table_cell_style), Paragraph("Emniyet Valfleri", table_cell_style), Paragraph("V-10 ve V-12 valfleri manuel test kolu ile blöf ettirilir.", table_cell_style), Paragraph("Bakım Teknisyeni", table_cell_style)],
        [Paragraph("Aylık", table_cell_style), Paragraph("Brülör ve Elektrotlar", table_cell_style), Paragraph("Ateşleme elektrot aralığı (3.5 mm) ayarlanır ve kurum temizlenir.", table_cell_style), Paragraph("Bakım Teknisyeni", table_cell_style)],
        [Paragraph("6 Aylık", table_cell_style), Paragraph("Kazan Taş ve Tortu Temizliği", table_cell_style), Paragraph("Kimyasal yıkama (Descaling) ile kireç birikintileri sökülür.", table_cell_style), Paragraph("Yetkili Servis", table_cell_style)],
        [Paragraph("Yıllık", table_cell_style), Paragraph("Tahribatsız Test (NDT)", table_cell_style), Paragraph("Kaynak dikişlerine ultrasonik çatlak testi uygulanır.", table_cell_style), Paragraph("Sertifikalı Mühendis", table_cell_style)]
    ]
    
    bakim_table = Table(bakim_data, colWidths=[70, 130, 220, 110])
    bakim_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EDF2F7")]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(bakim_table)

    # --- BÖLÜM 5: ELEKTRİK VE PLC BAĞLANTI İLKELERİ ---
    story.extend([
        Spacer(1, 10),
        Paragraph("5. Elektrik ve Otomasyon Bağlantı Kuralları", h1_style),
        Paragraph(
            "Kazan kontrol panosu 380V AC 50Hz trifaze şebeke gerilimi ile beslenmektedir. "
            "PLC giriş/çıkış modülleri 24V DC izole güç kaynağı üzerinden beslenmelidir. "
            "Topraklama direnci maksimum <b>2 Ohm</b> seviyesinde olmalı, nötr ve toprak hatları kesinlikle birleştirilmemelidir (TN-S sistemi).",
            body_style
        ),
    ])

    # Document Build
    doc.build(story)
    print(f"PDF başarıyla oluşturuldu: {filename}")

if __name__ == "__main__":
    build_pdf()