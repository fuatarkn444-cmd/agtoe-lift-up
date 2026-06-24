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

# --- 1. SAYFA VE SABİT TANIMLAMALARI ---
st.set_page_config(page_title="LIFT-UP Kestirimci Bakım", page_icon=" ✈️ ", layout="wide")

MALZEMELER = {
    "Alüminyum 6061-T6": {"kc": 800, "c_taylor": 4.5e10}, 
    "Alüminyum 7075-T6": {"kc": 975, "c_taylor": 3.5e10},
    "Titanyum Ti-6Al-4V": {"kc": 2100, "c_taylor": 1.2e10}, 
    "Paslanmaz Çelik 304": {"kc": 2100, "c_taylor": 2.8e10},
    "17-4 PH Paslanmaz": {"kc": 2600, "c_taylor": 2.0e10}, 
    "AISI 4340 Alaşımlı Çelik": {"kc": 2700, "c_taylor": 1.8e10}
}

if 'ilk_giris' not in st.session_state: st.session_state.ilk_giris = True
if 'analiz_yapildi' not in st.session_state: st.session_state.analiz_yapildi = False
if 'sistem_verisi' not in st.session_state: st.session_state.sistem_verisi = None

@st.dialog(" ✈️  LIFT-UP Sistemine Hoş Geldiniz")
def rehber_dialog():
    st.markdown("""
    **Bu sistem, İstatistiksel Regresyon analizleri kullanarak parça üzerindeki kritik bölgelerin aşınma ufuklarını tahmin eder.**

    ###  🛠️  Nasıl Kullanılır?
    1. **Veri Giriş Yöntemi:** Sol menüden CMM dosya yükleme (PDF/Otonom) veya manuel giriş yöntemini seçin.
    2. **Genel Ayarlar:** İzlemek istediğiniz maksimum tolerans sınırını girin.
    3. **Eşleştirme:** Yüklediğiniz dosyadan analiz etmek istediğiniz kritik bölgeleri seçin ve CAM işleme süresini girin.
    4. **Analiz:** 'Kestirim Analizini Başlat' butonuna basın ve modelin çizdiği aşınma grafiklerini inceleyin.
    """)

if st.session_state.ilk_giris:
    rehber_dialog()
    st.session_state.ilk_giris = False

# --- 2. CSS TEMA VE REMOVE BEFORE FLIGHT ŞERİDİ ---
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
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️  LIFT-UP: Kestirimci Bakım Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 15px; font-weight: bold; font-style: italic; letter-spacing: 1px;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

with col_logo:
    if os.path.exists("agtoe.png"): st.image("agtoe.png", width=150)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", width=150)

# --- 3. OTONOM TAKIM DEĞİŞİMİ ALGILAYICI MOTOR ---
def filtrele_ve_takim_degisimini_bul(raw_blocks, raw_vals):
    """CMM verilerindeki ani düşüşleri analiz ederek takım sıfırlanmasını tespit eder."""
    if not raw_vals: return [], []
    guncel_vals = [raw_vals[0]]
    guncel_blocks = [raw_blocks[0]]
    
    for i in range(1, len(raw_vals)):
        # Eğer 0.02 mm'den fazla ani bir düşüş varsa (Takım sökülüp yenisi takılmış demektir)
        if raw_vals[i-1] - raw_vals[i] > 0.02:
            guncel_vals = [raw_vals[i]]       # Seriyi sıfırla ve yeni takımdan başla
            guncel_blocks = [raw_blocks[i]]
        else:
            guncel_vals.append(raw_vals[i])
            guncel_blocks.append(raw_blocks[i])
    return guncel_blocks, guncel_vals

# --- 4. ZEISS CALYPSO PDF MOTORU ---
def extract_pdf_data_advanced(file):
    rows_list = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: continue
                for line in text.split('\n'):
                    line_clean = line.replace('"', '').strip()
                    if re.search(r'\d', line_clean) and not line_clean.startswith(("Name", "CMM", "Date", "Run", "Part")):
                        nums = re.findall(r'[-+]?\d+[\.,]\d+', line_clean)
                        if len(nums) >= 2:
                            ilk_sayi_index = line_clean.find(nums[0])
                            olcum_adi = line_clean[:ilk_sayi_index].strip()
                            try:
                                meas = float(nums[0].replace(',', '.'))
                                dev = float(nums[-1].replace(',', '.'))
                                nom = float(nums[1].replace(',', '.')) if len(nums) > 1 else 0.0
                                if olcum_adi:
                                    rows_list.append({
                                        "Olcum_Adi": olcum_adi, "Olculen_Deger": meas,
                                        "Nominal": nom, "Sapma": abs(dev),
                                        "Ust_Tolerans": 0.100, "Alt_Tolerans": -0.100
                                    })
                            except ValueError: continue
    except Exception as e:
        st.error(f"PDF Okuma Hatası: {e}")
    return pd.DataFrame(rows_list)

def clean_cmm_data(df):
    sutun_haritasi = {
        'Name': 'Olcum_Adi', 'Dimension': 'Olcum_Adi', 'Measured value': 'Olculen_Deger',
        'Nominal value': 'Nominal', '+Tol': 'Ust_Tolerans', '-Tol': 'Alt_Tolerans', 'Deviation': 'Sapma'
    }
    df.rename(columns=sutun_haritasi, inplace=True)
    for col in ['Olculen_Deger', 'Nominal', 'Ust_Tolerans', 'Alt_Tolerans', 'Sapma']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.'), errors='coerce').fillna(0)
    return df

# --- 5. GELİŞMİŞ POLİNOMİYAL KESTİRİM MOTORU ---
class AI_ToolLife:
    def __init__(self, tolerance, birim_ad):
        self.tolerance = float(tolerance)
        self.birim_ad = birim_ad
        self.scenarios = {}

    def add_scenario(self, name, blocks, wear_data, cam_cycle_time=None):
        if len(blocks) < 2:
            raise ValueError(f"{name} analizi için en az 2 ardışık parça ölçümü gerekiyor.")
            
        X = np.array(blocks).reshape(-1, 1)
        y = np.array(wear_data)
        max_blok = max(blocks)
        veri_sayisi = len(blocks)
        
        # Takımın karakteristik doğasına (S-Eğrisi) uygun 3. derece regresyon modeli
        deg = 3 if len(blocks) >= 4 else 2
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        mse = mean_squared_error(y, model.predict(X_poly))
        rmse_val = np.sqrt(mse)
        
        # Kök Bulma İşlemi (Polinomun toleransı kestiği nokta)
        coeffs = np.polyfit(X.flatten(), y, deg)
        coeffs_tol = coeffs.copy()
        coeffs_tol[-1] -= self.tolerance
        roots = np.roots(coeffs_tol)
        
        valid_roots = [r.real for r in roots if np.isreal(r) and r.real > X[-1][0]]
        uzak_tahmin_uyarisi = False
        
        if valid_roots:
            exact_cross = min(valid_roots)
            grafik_son_blok = min(int(np.ceil(exact_cross)) + 2, 5000)
            if exact_cross > (max_blok * 5): uzak_tahmin_uyarisi = True
            
            guven_araligi_metni = f"{exact_cross:.1f} Adet Parça"
            uretim_metni = f"Kestirim modeline göre aktif kesici takımınız {exact_cross:.1f}. parçadan sonra maksimum tolerans sınırını aşacaktır."
            
            if cam_cycle_time and cam_cycle_time > 0:
                exact_time_minutes = exact_cross * cam_cycle_time
                dk = int(exact_time_minutes)
                sn = int(round((exact_time_minutes - dk) * 60))
                sure_araligi_metni = f"{dk} Dk {sn} Sn"
            else:
                sure_araligi_metni = "Süre Belirtilmedi"
        else:
            grafik_son_blok = int(max_blok * 1.5)
            guven_araligi_metni = f"{grafik_son_blok}+ Parça"
            sure_araligi_metni = f"{grafik_son_blok * (cam_cycle_time if cam_cycle_time else 0):.1f}+ Dk"
            uretim_metni = "Analiz ufku boyunca bu bölgede herhangi bir boyutsal risk gözlemlenmemiştir."
            
        future_blocks = np.arange(0, grafik_son_blok + 1).reshape(-1, 1)
        future_y = model.predict(poly.transform(future_blocks))
        
        self.scenarios[name] = {
            'b_raw': blocks, 'y_raw': wear_data, 'b_fut': future_blocks.flatten(), 'y_fut': future_y,
            'rmse_val': rmse_val, 'guven_araligi_metni': guven_araligi_metni, 
            'sure_araligi_metni': sure_araligi_metni, 'uretim_metni': uretim_metni, 
            'cam_cycle_time': cam_cycle_time if cam_cycle_time else 0.0,
            'veri_sayisi': veri_sayisi, 'uzak_tahmin_uyarisi': uzak_tahmin_uyarisi
        }

    # ÇİFT GRAFİK MOTORU (Parça Sayısı + Süre)
    def plot_single_scenario(self, name):
        data = self.scenarios[name]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        uyari_siniri = self.tolerance * 0.75
        
        for ax, x_raw, x_fut, xlabel in zip(
            [ax1, ax2], 
            [data['b_raw'], [b * data['cam_cycle_time'] for b in data['b_raw']]],
            [data['b_fut'], [b * data['cam_cycle_time'] for b in data['b_fut']]],
            ["Ardışık İşlenen Parça Sırası (Adet)", "Toplam Aktif Kesme Süresi (Dakika)"]
        ):
            # Endüstriyel Renk Şeritleri
            ax.axhspan(0, uyari_siniri, facecolor='#d4edda', alpha=0.6, label='Güvenli İşleme Alanı (Yeşil)')
            ax.axhspan(uyari_siniri, self.tolerance, facecolor='#fff3cd', alpha=0.7, label='Erken Uyarı Alanı (Sarı)')
            ax.axhspan(self.tolerance, self.tolerance * 1.5, facecolor='#f8d7da', alpha=0.6, label='Boyutsal Risk Alanı (Kırmızı)')

            # Sınır Eşikleri
            ax.axhline(self.tolerance, color='#dc3545', linewidth=3, linestyle='--', label=f"Maksimum Tolerans ({self.tolerance} {self.birim_ad})")
            ax.axhline(uyari_siniri, color='#ffc107', linewidth=2.5, linestyle='--', label=f"Erken Uyarı Sınırı ({uyari_siniri} {self.birim_ad})")

            # Gerçekleşen Noktalar ve Gelişmiş Regresyon Eğrisi
            ax.plot(x_fut, data['y_fut'], color='#004B87', linestyle='-', linewidth=3.5, zorder=4, label="Kestirim Trend Eğrisi")
            ax.scatter(x_raw, data['y_raw'], color='#E31837', s=140, zorder=5, edgecolor='white', linewidth=1.5, label='Güncel CMM Ölçümleri')
            
            ax.set_title(f"Kritik Bölge Dashboard: {name.upper()}", fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
            ax.set_ylabel(f"Boyutsal Sapma [{self.birim_ad}]", fontsize=10, fontweight='bold')
            ax.set_ylim(0.0, self.tolerance * 1.4)
            if len(x_fut) > 0: ax.set_xlim(0, x_fut[-1])
            
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        fig.tight_layout(pad=3.0)
        return fig 

# --- 6. YAN MENÜ (SİDEBAR) ---
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

# --- 7. OTONOM DOSYA YÜKLEME ALGORİTMASI ---
df_ana = pd.DataFrame()
if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
    st.info("💡 **Otonom Mod:** CMM cihazından sırayla aldığınız ardışık PDF raporlarını (K1 S001, K1 S002 vb.) doğrudan buraya yükleyin.")
    yuklenen_dosyalar = st.file_uploader("CMM Raporlarını Klasör Halinde Sürükleyin", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
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

# --- 8. PARAMETRE GİRİŞİ VE EŞLEŞTİRME ---
st.markdown(f"###  📋  Kritik İzleme Bölgelerinin Yapılandırılması ({senaryo_sayisi} Bölge)")
sekmeler = st.tabs([f"{i+1}. Kritik Bölge" for i in range(senaryo_sayisi)])
senaryo_verileri = []
eksik_alanlar = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim = st.text_input(f"İzlenecek Bölge İsmi", value=f"Kritik Bölge {i+1}", key=f"isim_{i}")
        colA, colB, colC = st.columns([1.3, 1.3, 1.3])

        with colA:
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
                st.markdown("**CMM Rapor Eşleştirmesi**")
                if not df_ana.empty and 'Olcum_Adi' in df_ana.columns:
                    olcumler = df_ana['Olcum_Adi'].unique()
                    secilen_olcum = st.selectbox("📏 CMM Dosyasındaki Karşılığı", olcumler, key=f"geo_{i}", index=None, placeholder="Bölge Seçin...")
                    if not secilen_olcum: eksik_alanlar.append(f"{isim}: Geometri Seçimi")
                    aykiri_filtre = st.checkbox("Hatalı Ölçümleri (Outlier) Temizle", value=True, key=f"outlier_{i}")
                    cmm_str = ""
                else:
                    st.info("Geçerli bir CMM dosyası/PDF bekleniyor.")
                    eksik_alanlar.append(f"{isim}: CMM Dosyası")
                    secilen_olcum, aykiri_filtre, cmm_str = None, False, ""
            else:
                st.markdown("**Manuel Ölçüm Girişi**")
                cmm_str = st.text_input(f"Bölge Aşınma Değerleri ({birim_ad}, Boşluklu)", key=f"cmm_{i}", placeholder=cmm_ornek)
                if not cmm_str: eksik_alanlar.append(f"{isim}: CMM Verileri")
                secilen_olcum, aykiri_filtre = None, False

        with colB:
            st.markdown("**CAM Kesme ve Çevrim Verileri**")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                cam_dk = st.number_input("Çevrim Dakikası", min_value=0, key=f"cam_dk_{i}", value=None, placeholder="Örn: 2")
                if cam_dk is None: eksik_alanlar.append(f"{isim}: İşleme Süresi")
            with t_col2:
                cam_sn = st.number_input("Çevrim Saniyesi", min_value=0, max_value=59, key=f"cam_sn_{i}", value=None, placeholder="Örn: 15")
            
            cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None

        with colC:
            st.markdown("** 🧠 Arka Plan Matematiği**")
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)" and secilen_olcum:
                df_sub = df_ana[df_ana['Olcum_Adi'] == secilen_olcum]
                if not df_sub.empty and 'Sapma' in df_sub.columns:
                    st.success(f"✅ **{secilen_olcum}** bölgesi için {len(df_sub)} adet ardışık veri çekildi.")
            
            st.markdown("""
            <div style='background-color:rgba(248, 249, 250, 0.05); padding:10px; border-radius:5px; font-size:12px; border-left: 3px solid #004B87; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);'>
            Sistem, CMM verilerindeki ani düşüşleri algılayarak <b>takım değişimlerini otonom tespit eder.</b> Sadece aktif takımın aşınma verisi 3. derece regresyon ile modellenerek (başlangıç, kararlı aşınma, kırılma fazları) takımın toleransı aşacağı parça ve zaman öngörülür.
            </div>
            """, unsafe_allow_html=True)

        senaryo_verileri.append({
            "isim": isim, "cmm_str": cmm_str, "secilen_olcum": secilen_olcum, 
            "aykiri_filtre": aykiri_filtre, "cam_sure": cam_sure
        })

st.markdown("---")

if st.button(" 🚀  Kritik Bölge Kestirim Analizini Başlat", use_container_width=True, type="primary"):
    if tol_siniri is None: eksik_alanlar.append("Genel Ayarlar: Tolerans Limiti")
    
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
                    blocks, cmm_vals = filtrele_ve_takim_degisimini_bul(list(range(1, len(raw_vals) + 1)), raw_vals)
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
                    raw_blocks = df_sub['Parca_Sira'].tolist() if 'Parca_Sira' in df_sub.columns else list(range(1, len(raw_vals) + 1))
                    
                    # TAKIM DEĞİŞİMİ ALGORİTMASI BURADA ÇALIŞIYOR
                    blocks, cmm_vals = filtrele_ve_takim_degisimini_bul(raw_blocks, raw_vals)

                system.add_scenario(d["isim"], blocks, cmm_vals, d["cam_sure"])
            
            st.session_state.sistem_verisi = system
            st.session_state.analiz_yapildi = True

        except Exception as e:
            st.error(f"Sayısal modelleme hatası: Eğri çizimi için ardışık dosya serisinde yeterli veri yok. ({e})")
            st.session_state.analiz_yapildi = False

# --- 9. SEKMELİ ÇİFT GRAFİK VE RAPORLAMA ---
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
            col1.metric("Kestirim Güvenli Ömür Sınırı", veri['guven_araligi_metni'])
            col2.metric("Kestirim Ömür Zaman Hesabı", veri['sure_araligi_metni'])

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
