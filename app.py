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
    **Bu sistem, İstatistiksel Regresyon ve Taylor Denklemlerini harmanlayarak takım ömrünü kestirimci olarak tahmin eder.**

    ###  🛠️  Nasıl Kullanılır?
    1. **Veri Giriş Yöntemi:** Sol menüden CMM dosya yükleme (PDF/Excel/CSV) veya manuel giriş yöntemini seçin.
    2. **Parametreler:** Kullanacağınız malzeme, takım ölçüleri ve CAM kesme verilerini eksiksiz doldurun.
    3. **Eşleştirme:** Yüklediğiniz dosyadaki ölçüm geometrilerini ilgili kesici takımlarla eşleştirin.
    4. **Analiz:** 'Tahmini Başlat' butonuna basın ve modelin hesapladığı aşınma tahminlerini inceleyin.
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

col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️  LIFT-UP: Kestirimci Bakım Sistemi</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 15px; font-weight: bold; font-style: italic; letter-spacing: 1px;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

with col_logo:
    if os.path.exists("agtoe.png"): st.image("agtoe.png", width=150)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", width=150)

# --- ZEISS CALYPSO PDF METİN OKUMA MOTORU ---
def extract_pdf_data_advanced(file):
    rows_list = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')
                for line in lines:
                    cleaned_line = line.replace('"', '').strip()
                    parts = [p.strip() for p in cleaned_line.split(',') if p.strip()]
                    
                    if len(parts) >= 2:
                        name_candidate = parts[0]
                        if name_candidate in ["Name", "CMM Type", "CMM No", "Operator", "Programmer", "Date - Time", "Run"]:
                            continue
                        
                        measured_value_str = parts[1].replace('mm', '').strip()
                        deviation_str = parts[-1].replace('mm', '').strip() if len(parts) >= 5 else "0.0"
                        nominal_str = parts[2].replace('mm', '').strip() if len(parts) >= 3 else "0.0"
                        
                        try:
                            measured_value = float(measured_value_str.replace('°', ''))
                            deviation = float(deviation_str.replace('°', ''))
                            nominal = float(nominal_str.replace('°', ''))
                            
                            rows_list.append({
                                "Olcum_Adi": name_candidate,
                                "Olculen_Deger": measured_value,
                                "Nominal": nominal,
                                "Sapma": abs(deviation), 
                                "Ust_Tolerans": 0.100, 
                                "Alt_Tolerans": -0.100
                            })
                        except ValueError:
                            continue 
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

# --- KESTİRİMCİ MOTOR SINIFI ---
class AI_ToolLife:
    def __init__(self, tolerance, birim_ad):
        self.tolerance = float(tolerance)
        self.birim_ad = birim_ad
        self.scenarios = {}

    def calculate_daf(self, ae, D):
        if ae >= D: return 1.8
        elif ae <= (0.1 * D): return 1.1
        else: return 1.4

    def add_scenario(self, name, mat_name, kc_ref, c_taylor, D, z, Lc, Vc, fz, ap, ae, blocks, wear_data, cam_cycle_time):
        daf = self.calculate_daf(ae, D)
        hm = fz * (np.pi * D / (2 * z)) * (ae / D)**2 if ae >= D else fz * np.sqrt(ae / D)
        kc_eff = kc_ref * (hm / 0.1)**-0.2
        t_theo = (c_taylor / (Vc**3.5)) * (1 / (daf**1.5))
        
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
            
            exact_time_minutes = exact_cross * cam_cycle_time
            dk = int(exact_time_minutes)
            sn = int(round((exact_time_minutes - dk) * 60))
            if sn == 60:
                dk += 1
                sn = 0
            
            oran = exact_time_minutes / t_theo
            if oran >= 1.0: karsilastirma_durumu = "hata_buyuk"
            elif oran >= 0.75: karsilastirma_durumu = "tebrikler"
            elif oran <= 0.15: karsilastirma_durumu = "hata_kucuk"
            
            guven_araligi_metni = f"{exact_cross:.2f} Blok"
            uretim_metni = f"{exact_cross:.2f}. blokta aşınır."
            sure_araligi_metni = f"{dk} Dk {sn} Sn"
        else:
            grafik_son_blok = int(max_blok * 1.5)
            guven_araligi_metni = f"{grafik_son_blok}+ Blok"
            sure_araligi_metni = f"{grafik_son_blok * cam_cycle_time:.1f}+ Dk"
            uretim_metni = "Analiz ufku boyunca takımda riskli aşınma gözlemlenmemiştir."
            
        future_blocks = np.arange(1, grafik_son_blok + 1).reshape(-1, 1)
        future_y = model.predict(poly.transform(future_blocks))
        
        self.scenarios[name] = {
            'mat_name': mat_name, 'b_raw': blocks, 'y_raw': wear_data,
            'b_fut': future_blocks.flatten(), 'y_fut': future_y,
            'rmse_val': rmse_val, 't_theo': t_theo,
            'guven_araligi_metni': guven_araligi_metni,
            'sure_araligi_metni': sure_araligi_metni,
            'uretim_metni': uretim_metni,
            'cam_cycle_time': cam_cycle_time,
            'veri_sayisi': veri_sayisi,
            'uzak_tahmin_uyarisi': uzak_tahmin_uyarisi,
            'karsilastirma_durumu': karsilastirma_durumu,
            'D': D, 'z': z, 'Lc': Lc, 'Vc': Vc, 'fz': fz, 'ap': ap, 'ae': ae 
        }

    # --- ENDÜSTRİYEL COĞRAFİ GRAFİK ALGORİTMASI ---
    def plot_single_scenario(self, name):
        data = self.scenarios[name]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        uyari_siniri = self.tolerance * 0.75
        
        ax.axhspan(0, uyari_siniri, facecolor='#d4edda', alpha=0.4, label='Güvenli İşleme Alanı (Yeşil)')
        ax.axhspan(uyari_siniri, self.tolerance, facecolor='#fff3cd', alpha=0.5, label='Kritik Uyarı Alanı (Sarı)')
        ax.axhspan(self.tolerance, self.tolerance * 1.5, facecolor='#f8d7da', alpha=0.5, label='Boyutsal Risk Alanı (Kırmızı)')

        ax.axhline(self.tolerance, color='#dc3545', linewidth=2.5, linestyle='--', label=f"Maksimum Tolerans Limiti ({self.tolerance} {self.birim_ad})")
        ax.axhline(uyari_siniri, color='#ffc107', linewidth=2, linestyle='--', label=f"Erken Uyarı Eşiği ({uyari_siniri} {self.birim_ad})")

        ax.scatter(data['b_raw'], data['y_raw'], color='#004B87', s=130, zorder=5, edgecolor='white', linewidth=1.5, label='Raporlanan Parça Sapması')
        ax.plot(data['b_fut'], data['y_fut'], color='#17a2b8', linestyle='-', linewidth=3.5, zorder=4, label=f"Aşınma Regresyon Eğrisi")
        
        ax.set_title(f"Takım / Operasyon Kestirim Dashboard: {name.upper()}", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Üretilen Ardışık Parça Sırası (Blok Sayısı)", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"CMM Boyutsal Ölçüm Sapması [{self.birim_ad}]", fontsize=10, fontweight='bold')
        ax.set_ylim(0.0, self.tolerance * 1.5)
        ax.set_xlim(0, data['b_fut'][-1])
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        fig.tight_layout()
        return fig 

MALZEMELER = {
    "Alüminyum 6061-T6": {"kc": 800, "c_taylor": 4.5e10}, "Alüminyum 7075-T6": {"kc": 975, "c_taylor": 3.5e10},
    "Titanyum Ti-6Al-4V": {"kc": 2100, "c_taylor": 1.2e10}, "Paslanmaz Çelik 304": {"kc": 2100, "c_taylor": 2.8e10},
    "17-4 PH Paslanmaz": {"kc": 2600, "c_taylor": 2.0e10}, "AISI 4340 Alaşımlı Çelik": {"kc": 2700, "c_taylor": 1.8e10}
}

with st.sidebar:
    st.header(" ⚙️  Veri Giriş Sihirbazı")
    veri_giris_modu = st.radio("Sistemi Nasıl Kullanacaksınız?", ["CMM Dosyası Yükle (PDF/Otonom)", "Manuel Veri Girişi (Klasik)"])
    
    st.markdown("---")
    st.header(" 📏  Genel Ayarlar")
    birim_secimi = st.radio("Ölçüm Birimi Sistemi", ["Mikron (µm)", "Milimetre (mm)"], horizontal=True)
    is_mikron = "Mikron" in birim_secimi
    birim_ad = "Mikron" if is_mikron else "mm"
    
    tol_ornek = "Örn: 5" if is_mikron else "Örn: 0.005"
    cmm_ornek = "Örn: 0.2 0.5 0.8 1.2" if is_mikron else "Örn: 0.0002 0.0005 0.0008 0.0012"

    tol_siniri = st.number_input(f"Maksimum Tolerans ({birim_ad})", value=None, format="%g", placeholder=tol_ornek)
    senaryo_sayisi = st.number_input("Karşılaştırılacak Takım Sayısı", min_value=1, max_value=5, value=1, step=1)
    
    st.markdown("---")
    ortak_malzeme = st.checkbox("Tüm senaryolarda ortak MALZEME kullan", value=True)
    genel_malzeme_secimi = st.selectbox("Ortak Hammadde Seçimi", list(MALZEMELER.keys()), index=None, placeholder="Seçiniz...") if ortak_malzeme else None
    
    ortak_takim = st.checkbox("Tüm senaryolarda ortak TAKIM kullan", value=True)
    genel_t_cap = st.number_input("Ortak Takım Çapı (D) [mm]", value=None, min_value=1, placeholder="Örn: 6") if ortak_takim else None
    genel_t_dis = st.number_input("Ortak Takım Diş Sayısı (z)", value=None, min_value=1, placeholder="Örn: 4") if ortak_takim else None
    genel_t_boy = st.number_input("Ortak Takım Kesme Boyu (Lc) [mm]", value=None, min_value=1, placeholder="Örn: 24") if ortak_takim else None

df_ana = pd.DataFrame()
if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
    st.info("💡 **Otonom Mod:** CMM cihazınızdan aldığınız ardışık PDF raporlarını (K1 S001, K1 S002 vb.) veya Excel dosyalarını doğrudan buraya yükleyin.")
    
    df_sablon = pd.DataFrame({
        "Parça No": [1, 2, 3, 1, 2, 3],
        "Name": ["1_PROFILE", "1_PROFILE", "1_PROFILE", "4_02Position^2", "4_02Position^2", "4_02Position^2"],
        "Measured value": [0.053, 0.069, 0.088, 0.025, 0.039, 0.051],
        "Nominal value": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "+Tol": [0.100, 0.100, 0.100, 0.150, 0.150, 0.150],
        "-Tol": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "Deviation": [0.053, 0.069, 0.088, 0.025, 0.039, 0.051]
    })
    excel_sablon_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_sablon_buffer, engine='openpyxl') as writer:
        df_sablon.to_excel(writer, index=False, sheet_name='CMM_Data')
    st.download_button("📄 Örnek CMM Rapor Şablonunu İndir (.xlsx)", data=excel_sablon_buffer.getvalue(), file_name="TOMTAS_Ornek_CMM.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    yuklenen_dosyalar = st.file_uploader("CMM Raporlarını Sürükleyin (PDF, Excel, CSV)", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if yuklenen_dosyalar:
        veri_listesi = []
        parca_sayaci = 1
        for dosya in yuklenen_dosyalar:
            try:
                if dosya.name.lower().endswith('.pdf'):
                    df_temp = extract_pdf_data_advanced(dosya)
                    if not df_temp.empty:
                        df_temp['Parca_Sira'] = parca_sayaci
                elif dosya.name.lower().endswith('.csv'):
                    df_temp = pd.read_csv(dosya, sep=None, engine='python', decimal='.')
                    df_temp = clean_cmm_data(df_temp)
                else:
                    df_temp = pd.read_excel(dosya)
                    df_temp = clean_cmm_data(df_temp)
                
                if not df_temp.empty:
                    if 'Parca_Sira' not in df_temp.columns:
                        df_temp['Parca_Sira'] = parca_sayaci
                    veri_listesi.append(df_temp)
                    parca_sayaci += 1
            except Exception as e:
                st.error(f"Dosya okuma hatası ({dosya.name}): {e}")
                
        if veri_listesi:
            df_ana = pd.concat(veri_listesi, ignore_index=True)
            if 'Parca_Sira' in df_ana.columns and 'Olcum_Adi' in df_ana.columns: 
                df_ana = df_ana.sort_values(by=['Olcum_Adi', 'Parca_Sira'])

st.markdown(f"###  📋  Parametre Girişi ve Eşleştirme ({senaryo_sayisi} Takım)")
sekmeler = st.tabs([f"{i+1}. Takım" for i in range(senaryo_sayisi)])
senaryo_verileri = []
eksik_alanlar = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim = st.text_input(f"Takım İsmi (Örn: T01 - Çap Freze)", value=f"Takım {i+1}", key=f"isim_{i}")
        colA, colB, colC = st.columns([1.3, 1.3, 1])

        with colA:
            st.markdown("**Malzeme ve Takım Ayarları**")
            m_secim = genel_malzeme_secimi if ortak_malzeme else st.selectbox("Hammadde Seçimi", list(MALZEMELER.keys()), key=f"mat_{i}", index=None, placeholder="Seçiniz...")
            s_malzeme = MALZEMELER.get(m_secim) if m_secim else None
            if not s_malzeme: eksik_alanlar.append(f"{isim}: Hammadde")

            t_cap = genel_t_cap if ortak_takim else st.number_input("Takım Çapı (D) [mm]", min_value=1, key=f"tcap_{i}", value=None, placeholder="Örn: 6")
            t_dis = genel_t_dis if ortak_takim else st.number_input("Diş Sayısı (z)", min_value=1, key=f"tdis_{i}", value=None, placeholder="Örn: 4")
            t_boy = genel_t_boy if ortak_takim else st.number_input("Kesme Boyu (Lc) [mm]", min_value=1, key=f"tboy_{i}", value=None, placeholder="Örn: 24")
            if t_cap is None or t_dis is None or t_boy is None: eksik_alanlar.append(f"{isim}: Takım Ölçüleri")

        with colB:
            st.markdown("**Kesme Parametreleri**")
            vc = st.number_input("Kesme Hızı (Vc) [m/min]", min_value=1, key=f"vc_{i}", value=None, placeholder="Örn: 400")
            fz = st.number_input("İlerleme (fz) [mm/diş]", format="%g", key=f"fz_{i}", value=None, placeholder="Örn: 0.08")
            ap = st.number_input("Eksenel Derinlik (ap) [mm]", format="%g", key=f"ap_{i}", value=None, placeholder="Örn: 5")
            ae = st.number_input("Radyal Derinlik (ae) [mm]", format="%g", key=f"ae_{i}", value=None, placeholder="Örn: 5")
            
            if vc is None or fz is None or ap is None or ae is None: eksik_alanlar.append(f"{isim}: Kesme Verileri")

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                cam_dk = st.number_input("Dakika", min_value=0, key=f"cam_dk_{i}", value=None, placeholder="Örn: 2")
                if cam_dk is None: eksik_alanlar.append(f"{isim}: İşleme Süresi")
            with t_col2:
                cam_sn = st.number_input("Saniye", min_value=0, max_value=59, key=f"cam_sn_{i}", value=None, placeholder="Örn: 15")
            
            cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None

            secilen_olcum = None
            aykiri_filtre = False
            cmm_str = ""
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
                if not df_ana.empty and 'Olcum_Adi' in df_ana.columns:
                    olcumler = df_ana['Olcum_Adi'].unique()
                    secilen_olcum = st.selectbox("📏 Geometri / Unsur Eşleştirme", olcumler, key=f"geo_{i}", index=None, placeholder="İşlenen Unsuru Seçin...")
                    if not secilen_olcum: eksik_alanlar.append(f"{isim}: Geometri Eşleştirmesi")
                    aykiri_filtre = st.checkbox("Aykırı Ölçümleri Filtrele", value=True, key=f"outlier_{i}")
                else:
                    st.info("Eşleştirme için geçerli bir CMM dosyası/PDF bekleniyor.")
            else:
                cmm_str = st.text_input(f"CMM Verileri ({birim_ad}, Boşluklu)", key=f"cmm_{i}", placeholder=cmm_ornek)
                if not cmm_str: eksik_alanlar.append(f"{isim}: CMM Verileri")

        with colC:
            st.markdown("** 🧠 CMM Sağlık Kontrolü & Motor**")
            if s_malzeme: st.info(f"**Kc:** {s_malzeme['kc']} MPa | **Taylor:** {s_malzeme['c_taylor']:.1e}")
            else: st.warning("Malzeme seçimi bekleniyor...")
            
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)" and secilen_olcum:
                df_sub = df_ana[df_ana['Olcum_Adi'] == secilen_olcum]
                
                if not df_sub.empty and 'Olculen_Deger' in df_sub.columns and 'Nominal' in df_sub.columns and 'Ust_Tolerans' in df_sub.columns and 'Alt_Tolerans' in df_sub.columns:
                    mean_val = df_sub['Olculen_Deger'].mean()
                    std_val = df_sub['Olculen_Deger'].std()
                    nom = df_sub['Nominal'].iloc[0]
                    usl = nom + df_sub['Ust_Tolerans'].iloc[0]
                    lsl = nom - df_sub['Alt_Tolerans'].iloc[0]
                    
                    if pd.notna(std_val) and std_val > 0: cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val))
                    else: cpk = 0.0
                    
                    in_spec = df_sub[(df_sub['Olculen_Deger'] <= usl) & (df_sub['Olculen_Deger'] >= lsl)]
                    in_spec_pct = len(in_spec) / len(df_sub) * 100
                    
                    st.markdown(f"""
                    <div style='background-color:rgba(248, 249, 250, 0.05); padding:10px; border-radius:5px; border-left: 3px solid #28a745; margin-bottom: 10px;'>
                    <b>📊 {secilen_olcum} Analizi:</b><br>
                    Tolerans İçi Oran: <b>%{in_spec_pct:.1f}</b><br>
                    Proses Yeterliliği (Cpk): <b>{cpk:.2f}</b>
                    </div>
                    """, unsafe_allow_html=True)
                elif not df_sub.empty and 'Sapma' in df_sub.columns:
                    st.success(f"✅ {secilen_olcum} unsuru için {len(df_sub)} adet ardışık parça sapma verisi PDF'ten başarıyla çekildi.")
            
            st.markdown("""
            <div style='background-color:rgba(248, 249, 250, 0.05); padding:10px; border-radius:5px; font-size:12px; border-left: 3px solid #004B87; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); margin-top: 10px;'>
            <b>Takım ve Parça Eşleştirme Matematiği:</b><br>
            Sistem, yüklediğiniz CMM dosyasından sadece sizin <b>seçtiğiniz spesifik geometriye</b> ait boyutsal sapmaları çeker. Bu sapma verileri, o geometriyi üretirken kullandığınız kesici takımın fiziksel CAM verileriyle (Taylor Denklemi) birleştirilir. Böylece her bir takım ucunun ne zaman tolerans dışına çıkacağı <b>bağımsız ve otonom</b> olarak hesaplanır.
            </div>
            """, unsafe_allow_html=True)

        senaryo_verileri.append({
            "isim": isim, "mat_isim": m_secim, "mat_data": s_malzeme,
            "t_cap": t_cap, "t_dis": t_dis, "t_boy": t_boy,
            "vc": vc, "fz": fz, "ap": ap, "ae": ae, "cam_sure": cam_sure, 
            "cmm_str": cmm_str, "secilen_olcum": secilen_olcum, "aykiri_filtre": aykiri_filtre
        })

st.markdown("---")

if st.button(" 🚀  Takım Ömrü Kestirim Analizini Başlat", use_container_width=True, type="primary"):
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
                    cmm_vals = [float(x.replace(',', '.')) for x in d["cmm_str"].split()]
                    blocks = list(range(1, len(cmm_vals) + 1))
                else:
                    df_sub = df_ana[df_ana['Olcum_Adi'] == d["secilen_olcum"]].copy()
                    
                    if 'Sapma' not in df_sub.columns and 'Olculen_Deger' in df_sub.columns and 'Nominal' in df_sub.columns:
                        df_sub['Sapma'] = np.abs(df_sub['Olculen_Deger'] - df_sub['Nominal'])
                    
                    df_sub['Sapma'] = pd.to_numeric(df_sub['Sapma'].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.'), errors='coerce').fillna(0).abs()
                    
                    if d["aykiri_filtre"]:
                        mean_val = df_sub['Sapma'].mean()
                        std_val = df_sub['Sapma'].std()
                        if std_val > 0:
                            z_scores = np.abs((df_sub['Sapma'] - mean_val) / std_val)
                            df_sub = df_sub[z_scores < 3]
                            
                    cmm_vals = df_sub['Sapma'].tolist()
                    blocks = df_sub['Parca_Sira'].tolist() if 'Parca_Sira' in df_sub.columns else list(range(1, len(cmm_vals) + 1))

                system.add_scenario(d["isim"], d["mat_isim"], d["mat_data"]['kc'], d["mat_data"]['c_taylor'], d["t_cap"], d["t_dis"], d["t_boy"], d["vc"], d["fz"], d["ap"], d["ae"], blocks, cmm_vals, d["cam_sure"])
            
            st.session_state.sistem_verisi = system
            st.session_state.analiz_yapildi = True

        except Exception as e:
            st.error(f"Sayısal modelleme hatası: Eğri çizimi için ardışık dosya serisinde yeterli veri yok. ({e})")
            st.session_state.analiz_yapildi = False

# --- SEKMELİ GRAFİK VE RAPORLAMA ---
if st.session_state.analiz_yapildi and st.session_state.sistem_verisi is not None:
    system = st.session_state.sistem_verisi
    
    st.markdown("### 📊 Takım Bazlı Kestirim Dashboard Alanları")
    sonuc_sekmeleri = st.tabs(list(system.scenarios.keys()))
    cizilen_grafikler = {}
    
    for idx, (isim, veri) in enumerate(system.scenarios.items()):
        with sonuc_sekmeleri[idx]:
            fig = system.plot_single_scenario(isim)
            st.pyplot(fig)
            cizilen_grafikler[isim] = fig 
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Teorik Takım Ömrü", f"{veri['t_theo']:.1f} Dakika")
            col2.metric("Kestirim Kırılma Ufku (Blok)", veri['guven_araligi_metni'])
            col3.metric("Kestirim Kırılma Ufku (Süre)", veri['sure_araligi_metni'])

            st.info(f" 🎯  **Tahmini Aşınma Sınırı:** {veri['uretim_metni']}")
            
            if veri['veri_sayisi'] < 3: st.error(" ⚠️  **Düşük Veri Yoğunluğu:** Kararlı tahmin için en az 3 parça raporu yüklenmelidir.")
            if veri['uzak_tahmin_uyarisi']: st.warning(" 🔭  **Aşırı Uzak Tahmin:** Uzun vadeli kestirimler sapma toleransını artırabilir.")

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

        t_theo_toplam_saniye = round(veri['t_theo'] * 60)
        t_theo_dakika = t_theo_toplam_saniye // 60
        t_theo_saniye = t_theo_toplam_saniye % 60
        temiz_t_theo_metni = f"{t_theo_dakika} Dakika {t_theo_saniye} Saniye" if t_theo_saniye > 0 else f"{t_theo_dakika} Dakika"

        rapor_verileri.append({
            "Senaryo Adı": isim,
            "Alaşım Bilgisi": veri['mat_name'],
            "Takım Çapı (D) [mm]": veri['D'],
            "Takım Diş Sayısı (z)": veri['z'],
            "Takım Kesme Boyu (Lc) [mm]": veri['Lc'],
            "Kesme Hızı (Vc) [m/min]": veri['Vc'],
            "İlerleme Hızı (fz) [mm/diş]": veri['fz'],
            "Eksenel Derinlik (ap) [mm]": veri['ap'],
            "Radyal Derinlik (ae) [mm]": veri['ae'],
            "Aktif CAM Süresi (Dk/Blok)": temiz_cam_sure_metni,
            "CMM Aşınma Veri Serisi": " - ".join([f"{x:.4f}" for x in veri['y_raw']]), 
            "Teorik Takım Ömrü": temiz_t_theo_metni,
            "Kestirim Kırılma Ufku (Blok)": veri['guven_araligi_metni'],
            "Kestirim Kırılma Ufku (Zaman)": veri['sure_araligi_metni'],
            "Model Regresyon Sapması (RMSE)": round(veri['rmse_val'], 4),
            "Tahmini Aşınma Noktası": veri['uretim_metni'].replace('*', '') 
        })
    
    df_rapor = pd.DataFrame(rapor_verileri)
    df_rapor.set_index("Senaryo Adı", inplace=True)
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
