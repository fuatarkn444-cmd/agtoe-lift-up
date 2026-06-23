import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os
import io          
import zipfile   
import pdfplumber
import re

st.set_page_config(page_title="LIFT-UP Kestirimci Bakım", page_icon=" ✈️ ", layout="wide")

# --- SABİT TANIMLAMALAR ---
MALZEMELER = {
    "Alüminyum 6061-T6": {"kc": 800, "c_taylor": 4.5e10}, 
    "Alüminyum 7075-T6": {"kc": 975, "c_taylor": 3.5e10},
    "Titanyum Ti-6Al-4V": {"kc": 2100, "c_taylor": 1.2e10}, 
    "Paslanmaz Çelik 304": {"kc": 2100, "c_taylor": 2.8e10},
    "17-4 PH Paslanmaz": {"kc": 2600, "c_taylor": 2.0e10}, 
    "AISI 4340 Alaşımlı Çelik": {"kc": 2700, "c_taylor": 1.8e10}
}

# --- KALICI HAFIZA (SESSION STATE) TANIMLAMALARI ---
if 'ilk_giris' not in st.session_state:
    st.session_state.ilk_giris = True
if 'analiz_yapildi' not in st.session_state:
    st.session_state.analiz_yapildi = False
if 'sistem_verisi' not in st.session_state:
    st.session_state.sistem_verisi = None

@st.dialog(" ✈️  LIFT-UP Sistemine Hoş Geldiniz")
def rehber_dialog():
    st.markdown("""
    **Bu sistem, İstatistiksel Regresyon analizleri kullanarak parça üzerindeki kritik bölgelerin aşınma ufuklarını tahmin eder.**

    ###  🛠️  Nasıl Kullanılır?
    1. **Veri Giriş Yöntemi:** Sol menüden CMM dosya yükleme (PDF/Otonom) veya manuel giriş yöntemini seçin.
    2. **Genel Ayarlar:** İzlemek istediğiniz maksimum tolerans sınırını girin.
    3. **Eşleştirme:** Yüklediğiniz dosyadan analiz etmek istediğiniz kritik bölgeleri seçin.
    4. **Analiz:** 'Kestirim Analizini Başlat' butonuna basın ve modelin çizdiği aşınma grafiklerini inceleyin.
    """)

if st.session_state.ilk_giris:
    rehber_dialog()
    st.session_state.ilk_giris = False

# --- CSS TEMA ---
st.markdown("""
<style>
header[data-testid="stHeader"] { background: linear-gradient(90deg, #004B87, #E31837) !important; height: 4px !important; }
h1, h2, h3, h4 { color: #004B87 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
[data-testid="metric-container"] { border: 1px solid rgba(136, 136, 136, 0.2); padding: 15px; border-radius: 8px; background-color: rgba(248, 249, 250, 0.05); box-shadow: 2px 2px 5px rgba(0,0,0,0.1); transition: transform 0.2s ease; }
[data-testid="metric-container"]:hover { transform: translateY(-3px); box-shadow: 3px 3px 8px rgba(0,0,0,0.15); }
[data-testid="stMetricValue"] { color: #004B87 !important; font-weight: 800; }
div.stButton > button:first-child { background: linear-gradient(90deg, #004B87, #0066cc); color: #FFFFFF; border: none; border-radius: 6px; font-weight: bold; padding: 10px 24px; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
div.stButton > button:first-child:hover { background: linear-gradient(90deg, #E31837, #ff3333); color: #FFFFFF; transform: scale(1.02); box-shadow: 0 6px 10px rgba(0,0,0,0.25); }
[data-testid="stSidebar"] { border-right: 3px solid #E31837; }
[data-testid="stSidebar"]::before { content: "REMOVE BEFORE FLIGHT"; display: block; background-color: #E31837; color: white; font-family: monospace; font-weight: bold; text-align: center; padding: 6px; letter-spacing: 1.5px; margin-bottom: 20px; border-radius: 0 0 5px 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: left; background-color: #E31837; color: white; display: inline-block; padding: 3px 12px; font-family: monospace; font-weight: bold; border-radius: 4px; font-size: 13px; box-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>by Fuat Arıkan</div>", unsafe_allow_html=True)

# --- HEADER PANEL ---
col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️  LIFT-UP: Kestirimci Bakım Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 15px; font-weight: bold; font-style: italic; letter-spacing: 1px;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

with col_logo:
    if os.path.exists("agtoe.png"): st.image("agtoe.png", width=150)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", width=150)

# --- EKİP ARKADAŞININ YAZDIĞI AKILLI VERİ DÜZELTME MOTORU (yapay_zeka_motoru.py'den Alındı) ---
def veriyi_duzelt_rolling_max(wear_verileri):
    """CMM ölçümlerindeki zikzakları ve prob hatalarını ayıklayıp kümülatif aşınma trendi üretir."""
    if not wear_verileri:
        return []
    wear = np.array(wear_verileri, dtype=float)
    islenmis = []
    current_max = wear[0]
    for w in wear:
        if w > current_max:
            current_max = w
        islenmis.append(current_max)
    return islenmis

# --- ZEISS CALYPSO PDF METİN OKUMA MOTORU ---
def extract_pdf_data_advanced(file):
    rows_list = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: continue
                lines = text.split('\n')
                for line in lines:
                    line_clean = line.strip()
                    match = re.search(r'^([a-zA-Z0-9_\^\s\u00d8]+)\s+([-+]?[0-9\.,/°\s]+mm|[-+]?[0-9\.,/°]+)', line_clean)
                    if match:
                        olcum_adi = match.group(1).strip()
                        if olcum_adi in ["Name", "CMM Type", "CMM No", "Operator", "Programmer", "Date - Time", "Run", "Part Number"]:
                            continue
                        numbers = re.findall(r'[-+]?[0-9\.,]+', line_clean.replace('mm', ''))
                        clean_numbers = []
                        for num in numbers:
                            try:
                                clean_numbers.append(float(num.replace(',', '.')))
                            except ValueError: continue
                        if len(clean_numbers) >= 2:
                            rows_list.append({
                                "Olcum_Adi": olcum_adi,
                                "Olculen_Deger": clean_numbers[0],
                                "Nominal": clean_numbers[1] if len(clean_numbers) > 2 else 0.0,
                                "Sapma": abs(clean_numbers[-1]),
                                "Ust_Tolerans": 0.100, 
                                "Alt_Tolerans": -0.100
                            })
    except Exception as e:
        st.error(f"PDF Analiz Motoru Hatası: {e}")
    return pd.DataFrame(rows_list)

def clean_cmm_data(df):
    sutun_haritasi = {
        'Name': 'Olcum_Adi', 'Dimension': 'Olcum_Adi', 'Olcum_Adi': 'Olcum_Adi',
        'Measured value': 'Olculen_Deger', 'Measurement': 'Olculen_Deger', 'Olculen_Deger': 'Olculen_Deger',
        'Nominal value': 'Nominal', 'Nominal': 'Nominal',
        '+Tol': 'Ust_Tolerans', 'Upper_Tolerance': 'Ust_Tolerans', 'Ust_Tolerans': 'Ust_Tolerans',
        '-Tol': 'Alt_Tolerans', 'Lower_Tolerance': 'Alt_Tolerans', 'Alt_Tolerans': 'Alt_Tolerans',
        'Deviation': 'Sapma', 'Sapma': 'Sapma', 'Parça No': 'Parca_Sira'
    }
    df.rename(columns=sutun_haritasi, inplace=True)
    for col_name in ['Olculen_Deger', 'Nominal', 'Ust_Tolerans', 'Alt_Tolerans', 'Sapma']:
        if col_name in df.columns and df[col_name].dtype == object:
            df[col_name] = df[col_name].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.').astype(float)
    return df

# --- KESTİRİMCİ FİZİK VE REGRESYON MOTORU ---
class AI_ToolLife:
    def __init__(self, tolerance, birim_ad):
        self.tolerance = float(tolerance)
        self.birim_ad = birim_ad
        self.scenarios = {}

    def add_scenario(self, name, blocks, wear_data, cam_cycle_time=None, t_theo=None, mat_name=None):
        X = np.array(blocks).reshape(-1, 1)
        y = np.array(wear_data)
        max_blok = max(blocks)
        veri_sayisi = len(blocks)
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        mse = mean_squared_error(y, model.predict(X_poly))
        rmse_val = np.sqrt(mse)
        
        a, b, c = model.coef_[2], model.coef_[1], model.intercept_ - self.tolerance
        coefs = [a, b, c] if abs(a) > 1e-15 else [b, c]
        roots = np.roots(coefs)
        valid_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
        
        karsilastirma_durumu = "normal"
        uzak_tahmin_uyarisi = False
        
        if valid_roots:
            raw_cross = min(valid_roots)
            exact_cross = round(raw_cross * 4) / 4
            if exact_cross <= 0: exact_cross = 0.25 
                
            grafik_son_blok = min(int(np.ceil(exact_cross)) + 2, 5000)
            if exact_cross > (max_blok * 5): uzak_tahmin_uyarisi = True
            
            guven_araligi_metni = f"{exact_cross:.2f} Adet Parça"
            uretim_metni = f"Kestirim modeline göre bu kritik bölge {exact_cross:.2f}. parçadan sonra maksimum tolerans sınırını aşacaktır."
            
            if cam_cycle_time:
                exact_time_minutes = exact_cross * cam_cycle_time
                dk = int(exact_time_minutes)
                sn = int(round((exact_time_minutes - dk) * 60))
                sure_araligi_metni = f"{dk} Dk {sn} Sn Çevrim Süresi"
                if t_theo:
                    oran = exact_time_minutes / t_theo
                    if oran >= 1.0: karsilastirma_durumu = "hata_buyuk"
                    elif oran >= 0.75: karsilastirma_durumu = "tebrikler"
                    elif oran <= 0.15: karsilastirma_durumu = "hata_kucuk"
            else:
                sure_araligi_metni = "Belirlenmedi (Manuel Mod Girdisi)"
        else:
            grafik_son_blok = int(max_blok * 1.5)
            guven_araligi_metni = f"{grafik_son_blok}+ Parça"
            sure_araligi_metni = "Risk Sınırı Dışında"
            uretim_metni = "Analiz ufku boyunca bu bölgede herhangi bir boyutsal risk gözlemlenmemiştir."
            
        future_blocks = np.arange(1, grafik_son_blok + 1).reshape(-1, 1)
        future_y = model.predict(poly.transform(future_blocks))
        
        self.scenarios[name] = {
            'b_raw': blocks, 'y_raw': wear_data,
            'b_fut': future_blocks.flatten(), 'y_fut': future_y,
            'rmse_val': rmse_val, 't_theo': t_theo if t_theo else 0.0,
            'guven_araligi_metni': guven_araligi_metni,
            'sure_araligi_metni': sure_araligi_metni,
            'uretim_metni': uretim_metni,
            'cam_cycle_time': cam_cycle_time if cam_cycle_time else 0.0,
            'veri_sayisi': veri_sayisi,
            'uzak_tahmin_uyarisi': uzak_tahmin_uyarisi,
            'karsilastirma_durumu': karsilastirma_durumu,
            'mat_name': mat_name if mat_name else "Belirlenmedi"
        }

    def plot_single_scenario(self, name):
        data = self.scenarios[name]
        fig, ax = plt.subplots(figsize=(10, 5))
        uyari_siniri = self.tolerance * 0.75
        
        # Üç Renkli Bölge Şeritleri (Görseliniz ile %100 Senkronize Düzen)
        ax.axhspan(0, uyari_siniri, facecolor='#d4edda', alpha=0.6, label='Güvenli İşleme Alanı (Yeşil)')
        ax.axhspan(uyari_siniri, self.tolerance, facecolor='#fff3cd', alpha=0.7, label='Erken Uyarı Alanı (Sarı)')
        ax.axhspan(self.tolerance, self.tolerance * 1.5, facecolor='#f8d7da', alpha=0.6, label='Boyutsal Risk Alanı (Kırmızı)')

        ax.axhline(self.tolerance, color='#dc3545', linewidth=3, linestyle='--', label=f"Maksimum Tolerans Limiti ({self.tolerance} {self.birim_ad})")
        ax.axhline(uyari_siniri, color='#ffc107', linewidth=2.5, linestyle='--', label=f"Erken Uyarı Sınırı ({uyari_siniri} {self.birim_ad})")

        ax.plot(data['b_fut'], data['y_fut'], color='#004B87', linestyle='-', linewidth=3.5, zorder=4, label="Aşınma Tahmin Eğrisi")
        ax.scatter(data['b_raw'], data['y_raw'], color='#E31837', s=140, zorder=5, edgecolor='white', linewidth=1.5, label='Kümülatif CMM Sapma Noktaları')
        
        ax.set_title(f"Kritik Bölge Kestirim Dashboard Analizi: {name.upper()}", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Ardışık Üretilen Parça Sırası (Adet)", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"CMM Boyutsal Ölçüm Sapması [{self.birim_ad}]", fontsize=10, fontweight='bold')
        ax.set_ylim(0.0, self.tolerance * 1.4)
        ax.set_xlim(1, data['b_fut'][-1])
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)
        fig.tight_layout()
        return fig 

# --- YAN MENÜ PANELİ ---
with st.sidebar:
    st.header(" ⚙️  Veri Giriş Sihirbazı")
    veri_giris_modu = st.radio("Sistemi Nasıl Kullanacaksınız?", ["CMM Dosyası Yükle (PDF/Otonom)", "Manuel Veri Girişi (Klasik)"])
    
    st.markdown("---")
    st.header(" 📏  Genel Ayarlar")
    birim_secimi = st.radio("Ölçüm Birimi Sistemi", ["Mikron (µm)", "Milimetre (mm)"], horizontal=True)
    is_mikron = "Mikron" in birim_secimi
    birim_ad = "Mikron" if is_mikron else "mm"
    
    tol_ornek = "Örn: 5" if is_mikron else "Örn: 0.05"
    cmm_ornek = "Örn: 0.2 0.5 0.8 1.2" if is_mikron else "Örn: 0.0002 0.0005 0.0008 0.0012"

    tol_siniri = st.number_input(f"Maksimum Tolerans Limiti ({birim_ad})", value=None, format="%g", placeholder=tol_ornek)
    senaryo_sayisi = st.number_input("İzlenecek Kritik Bölge Sayısı", min_value=1, max_value=5, value=1, step=1)
    
    # Sadece Manuel Modda Görünen Gelişmiş Seçenekler
    if veri_giris_modu == "Manuel Veri Girişi (Klasik)":
        st.markdown("---")
        st.subheader(" 📜 Manuel Mod Ek Parametreleri")
        ortak_malzeme = st.checkbox("Tüm operasyonlarda ortak hammadde kullan", value=True)
        genel_malzeme_secimi = st.selectbox("Ortak Hammadde Seçimi", list(MALZEMELER.keys()), index=None, placeholder="Seçiniz...") if ortak_malzeme else None

# --- DOSYA YÜKLEME VE HAVACILIK PARSER ALGORTİMASI ---
df_ana = pd.DataFrame()
if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
    st.info("💡 **Otonom Mod:** CMM cihazından sırayla aldığınız ardışık PDF raporlarını (K1 S001, K1 S002 vb.) doğrudan buraya sürükleyin. Kesici takım veya Taylor ömür verisi girmenize gerek yoktur.")
    
    df_sablon = pd.DataFrame({
        "Parça No": [1, 2, 3], "Name": ["1_PROFILE", "1_PROFILE", "1_PROFILE"],
        "Measured value": [0.053, 0.069, 0.088], "Nominal value": [0.000, 0.000, 0.000],
        "+Tol": [0.100, 0.100, 0.100], "-Tol": [0.000, 0.000, 0.000], "Deviation": [0.053, 0.069, 0.088]
    })
    excel_sablon_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_sablon_buffer, engine='openpyxl') as writer:
        df_sablon.to_excel(writer, index=False, sheet_name='CMM_Data')
    st.download_button("📄 Örnek CMM Rapor Şablonunu İndir (.xlsx)", data=excel_sablon_buffer.getvalue(), file_name="TOMTAS_Ornek_CMM.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    yuklenen_dosyalar = st.file_uploader("CMM Raporlarını Klasör Halinde Sürükleyin (PDF, Excel, CSV)", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if yuklenen_dosyalar:
        veri_listesi = []
        parca_sayaci = 1
        for dosya in yuklenen_dosyalar:
            try:
                if dosya.name.lower().endswith('.pdf'):
                    df_temp = extract_pdf_data_advanced(dosya)
                elif dosya.name.lower().endswith('.csv'):
                    df_temp = pd.read_csv(dosya, sep=None, engine='python', decimal='.')
                    df_temp = clean_cmm_data(df_temp)
                else:
                    df_temp = pd.read_excel(dosya)
                    df_temp = clean_cmm_data(df_temp)
                
                if not df_temp.empty:
                    df_temp['Parca_Sira'] = parca_sayaci
                    veri_listesi.append(df_temp)
                    parca_sayaci += 1
            except Exception as e:
                st.error(f"Dosya okuma hatası ({dosya.name}): {e}")
                
        if veri_listesi:
            df_ana = pd.concat(veri_listesi, ignore_index=True)
            if 'Olcum_Adi' in df_ana.columns: 
                df_ana = df_ana.sort_values(by=['Olcum_Adi', 'Parca_Sira'])

# --- REORGANİZE EDİLMİŞ BÖLGE ODAKLI SEKMELER ---
st.markdown(f"###  📋  Kritik İzleme Bölgelerinin Yapılandırılması ({senaryo_sayisi} Bölge)")
sekmeler = st.tabs([f"{i+1}. Kritik Bölge" for i in range(senaryo_sayisi)])
senaryo_verileri = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim_varsayilan = f"Kritik Bölge {i+1}"
        isim = st.text_input(f"İzlenecek Bölge / Unsur İsmi Adı", value=isim_varsayilan, key=f"isim_{i}")
        colA, colB = st.columns([2, 1])

        with colA:
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
                if not df_ana.empty and 'Olcum_Adi' in df_ana.columns:
                    olcumler = df_ana['Olcum_Adi'].unique()
                    secilen_olcum = st.selectbox(f"📏 Rapor Sütunlarındaki Karşılığı ({isim})", olcumler, key=f"geo_{i}", index=None, placeholder="CMM Bölgesi Seçin...")
                    if not secilen_olcum: eksik_alanlar.append(f"{isim}: Geometri Seçimi")
                    aykiri_filtre = st.checkbox("Hatalı Ölçümleri (Outlier) Temizle", value=True, key=f"outlier_{i}")
                    
                    # Manuel mod girdilerini otonomda sıfırlıyoruz
                    cmm_str, cam_sure, t_theo, m_secim = "", None, None, None
                else:
                    st.info("Eşleştirme listesi için geçerli bir CMM dosyası/PDF bekleniyor.")
                    eksik_alanlar.append(f"{isim}: CMM Dosyası")
                    secilen_olcum, aykiri_filtre, cmm_str, cam_sure, t_theo, m_secim = None, False, "", None, None, None
            else:
                # MANUEL MOD (KLASİK ESKİ YAPI) - İstendiği için korundu
                st.markdown("**Manuel CAM ve Taylor Parametreleri**")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    m_secim = genel_malzeme_secimi if ortak_malzeme else st.selectbox("Hammadde Seçimi", list(MALZEMELER.keys()), key=f"mat_{i}", index=None, placeholder="Seçiniz...")
                    s_malzeme = MALZEMELER.get(m_secim) if m_secim else None
                    vc = st.number_input("Kesme Hızı (Vc) [m/min]", min_value=1, key=f"vc_{i}", value=None, placeholder="Örn: 400")
                    fz = st.number_input("İlerleme (fz) [mm/diş]", format="%g", key=f"fz_{i}", value=None, placeholder="Örn: 0.08")
                    ap = st.number_input("Eksenel Derinlik (ap) [mm]", format="%g", key=f"ap_{i}", value=None, placeholder="Örn: 5")
                    ae = st.number_input("Radyal Derinlik (ae) [mm]", format="%g", key=f"ae_{i}", value=None, placeholder="Örn: 5")
                with col_m2:
                    t_cap = genel_t_cap if ortak_takim else st.number_input("Takım Çapı (D) [mm]", value=None, min_value=1, placeholder="Örn: 6")
                    t_dis = genel_t_dis if ortak_takim else st.number_input("Takım Diş Sayısı (z)", value=None, min_value=1, placeholder="Örn: 4")
                    cam_dk = st.number_input("Çevrim Dakikası", min_value=0, key=f"cam_dk_{i}", value=None, placeholder="Örn: 2")
                    cam_sn = st.number_input("Çevrim Saniyesi", min_value=0, max_value=59, key=f"cam_sn_{i}", value=None, placeholder="Örn: 15")
                
                cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None
                cmm_str = st.text_input(f"Kritik Bölge Ölçüm Değerleri ({birim_ad}, Boşluklu Geri)", key=f"cmm_{i}", placeholder=cmm_ornek)
                
                # Arka Planda Manuel Mod için Taylor Hesaplaması
                if s_malzeme and t_cap and t_dis and vc and fz and ae:
                    daf = 1.8 if ae >= t_cap else (1.1 if ae <= (0.1 * t_cap) else 1.4)
                    t_theo = (s_malzeme['c_taylor'] / (vc**3.5)) * (1 / (daf**1.5))
                else: t_theo = None
                
                if not cmm_str: eksik_alanlar.append(f"{isim}: Manuel CMM Serisi")
                secilen_olcum, aykiri_filtre = None, False

        with colB:
            st.markdown("** 🧠 Sağlık Kontrolü Paneli**")
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)" and secilen_olcum:
                df_sub = df_ana[df_ana['Olcum_Adi'] == secilen_olcum]
                if not df_sub.empty and 'Sapma' in df_sub.columns:
                    st.success(f"✅ **{secilen_olcum}** bölgesi için {len(df_sub)} adet ardışık parça aşınma serisi rapordan başarıyla çekildi.")
            else:
                st.warning("Manuel modda harici CMM sağlık kontrolü devre dışıdır.")
                
            st.markdown("""
            <div style='background-color:rgba(248, 249, 250, 0.05); padding:10px; border-radius:5px; font-size:12px; border-left: 3px solid #004B87; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); margin-top: 10px;'>
            <b>Esnek Bölge Analiz Mimarisi:</b><br>
            Sistem, Taylor takım ucu hesaplama zorunluluğunu ortadan kaldırarak doğrudan parça üzerindeki kritik bölgelerdeki sapmaları modeller. Bu sayede takım bilgisinden bağımsız olarak, sadece geometrik trend izlenerek aşınma kestirimi yapılır.
            </div>
            """, unsafe_allow_html=True)

        senaryo_verileri.append({
            "isim": isim, "cmm_str": cmm_str, "secilen_olcum": secilen_olcum, "aykiri_filtre": aykiri_filtre,
            "cam_sure": cam_sure, "t_theo": t_theo, "mat_isim": m_secim if m_secim else "Otonom Takip"
        })

st.markdown("---")

if st.button(" 🚀  Kritik Bölge Kestirim Analizini Başlat", use_container_width=True, type="primary"):
    if tol_siniri is None: eksik_alanlar.append("Genel Ayarlar: Tolerans")
    
    if len(eksik_alanlar) > 0:
        hata_metni = "\n".join([f"- {alan}" for alan in list(set(eksik_alanlar))])
        st.error(f" ⚠️  Lütfen analizi başlatmadan önce aşağıdaki eksik bilgileri doldurunuz:\n\n{hata_metni}")
        st.session_state.analiz_yapildi = False
    else:
        try:
            system = AI_ToolLife(tolerance=tol_siniri, birim_ad=birim_ad)
            for d in senaryo_verileri:
                if veri_giris_modu == "Manuel Veri Girişi (Klasik)":
                    raw_vals = [float(x.replace(',', '.')) for x in d["cmm_str"].split()]
                    cmm_vals = veriyi_duzelt_rolling_max(raw_vals)
                    blocks = list(range(1, len(cmm_vals) + 1))
                else:
                    df_sub = df_ana[df_ana['Olcum_Adi'] == d["secilen_olcum"]].copy()
                    df_sub['Sapma'] = pd.to_numeric(df_sub['Sapma'].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.'), errors='coerce').fillna(0).abs()
                    
                    if d["aykiri_filtre"]:
                        mean_val = df_sub['Sapma'].mean()
                        std_val = df_sub['Sapma'].std()
                        if std_val > 0:
                            z_scores = np.abs((df_sub['Sapma'] - mean_val) / std_val)
                            df_sub = df_sub[z_scores < 3]
                            
                    raw_vals = df_sub['Sapma'].tolist()
                    cmm_vals = veriyi_duzelt_rolling_max(raw_vals)
                    blocks = list(range(1, len(cmm_vals) + 1))

                system.add_scenario(d["isim"], blocks, cmm_vals, d["cam_sure"], d["t_theo"], d["mat_isim"])
            
            st.session_state.sistem_verisi = system
            st.session_state.analiz_yapildi = True

        except Exception as e:
            st.error(f"Sayısal modelleme hatası: Eğri çizimi için yeterli parça serisi verisi yok. ({e})")
            st.session_state.analiz_yapildi = False

# --- SEKMELİ GRAFİK VE RAPORLAMA ÇIKTILARI ---
if st.session_state.analiz_yapildi and st.session_state.sistem_verisi is not None:
    system = st.session_state.sistem_verisi
    
    st.markdown("### 📊 Bölge Bazlı Kestirim Dashboard Alanları")
    sonuc_sekmeleri = st.tabs(list(system.scenarios.keys()))
    cizilen_grafikler = {}
    
    for idx, (isim, veri) in enumerate(system.scenarios.items()):
        with sonuc_sekmeleri[idx]:
            fig = system.plot_single_scenario(isim)
            st.pyplot(fig)
            cizilen_grafikler[isim] = fig 
            
            col1, col2 = st.columns(2)
            col1.metric("Kestirim Güvenli Ömür Sınırı (Parça)", veri['guven_araligi_metni'].split()[0] + " Adet")
            col2.metric("Kestirim Ömür Zaman Hesabı", veri['sure_araligi_metni'].replace('Toplam Kesme Süresi', ''))

            st.info(f" 🎯  **Kestirim Raporu Metni:** {veri['uretim_metni']}")
            
            if veri['veri_sayisi'] < 3: st.error(" ⚠️  **Düşük Veri Yoğunluğu:** Kararlı tahmin için en az 3 ardışık parça verisi yüklenmelidir.")
            if veri['uzak_tahmin_uyarisi']: st.warning(" 🔭  **Aşırı Uzak Tahmin:** Uzun vadeli eğri kestirimleri sapma toleransını artırabilir.")

    st.markdown("---")
    st.subheader(" 📦 Tüm Analiz Paketini Kaydet")
    
    dosya_ismi_girdisi = st.text_input("📁 İndirilecek Rapor Klasörü Adı:", value="TOMTAS_LIFTUP_Rapor")
    temiz_isim = dosya_ismi_girdisi.strip() if dosya_ismi_girdisi.strip() else "TOMTAS_LIFTUP_Rapor"
        
    zip_isim = f"{temiz_isim}.zip"
    excel_isim = f"{temiz_isim}_Kiyaslama_Matrisi.xlsx"
    
    rapor_verileri = []
    for isim, veri in system.scenarios.items():
        toplam_saniye = round(veri['cam_cycle_time'] * 60)
        m_dakika = toplam_saniye // 60
        s_saniye = toplam_saniye % 60
        temiz_cam_sure_metni = f"{m_dakika} Dk {s_saniye} Sn" if s_saniye > 0 else f"{m_dakika} Dk"

        rapor_verileri.append({
            "Kritik Bölge / Unsur Adı": isim, 
            "Hammadde Durumu": veri['mat_name'],
            "Çevrim Süresi (Parça Başı)": temiz_cam_sure_metni if veri['cam_cycle_time'] > 0 else "Manuel Mod",
            "CMM Kümülatif Aşınma Serisi": " - ".join([f"{x:.4f}" for x in veri['y_raw']]), 
            "Kestirim Kırılma Ufku (Parça)": veri['guven_araligi_metni'], 
            "Kestirim Kırılma Ufku (Zaman)": veri['sure_araligi_metni'],
            "Model Regresyon Sapması (RMSE)": round(veri['rmse_val'], 4), 
            "Tahmini Aşınma Özeti": veri['uretim_metni']
        })
    
    df_rapor = pd.DataFrame(rapor_verileri)
    df_rapor.set_index("Kritik Bölge / Unsur Adı", inplace=True)
    df_rapor_final = df_rapor.T.reset_index()
    df_rapor_final.rename(columns={'index': 'Parametreler ve Sonuçlar'}, inplace=True)
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_rapor_final.to_excel(writer, index=False, sheet_name='Kestirim Raporu')
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(excel_isim, excel_buffer.getvalue())
        for g_isim, f in cizilen_grafikler.items():
            img_buffer = io.BytesIO()
            f.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300) 
            zip_file.writestr(f"{temiz_isim}_{g_isim}_Egri_Grafik.png", img_buffer.getvalue())

    st.download_button(
        label="📥 " + zip_isim + " Paketini İndir (Excel + Sektörel Grafikler)",
        data=zip_buffer.getvalue(),
        file_name=zip_isim,
        mime="application/zip",
        use_container_width=True
    )
