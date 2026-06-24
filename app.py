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

# --- 2. CSS TEMA VE REMOVE BEFORE FLIGHT ŞERİDİ ---
st.markdown("""
<style>
header[data-testid="stHeader"] { background: linear-gradient(90deg, #004B87, #E31837) !important; height: 4px !important; }
h1, h2, h3, h4 { color: #004B87 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
[data-testid="metric-container"] { border: 1px solid rgba(136, 136, 136, 0.2); padding: 15px; border-radius: 8px; background-color: rgba(248, 249, 250, 0.05); box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
[data-testid="metric-container"]:hover { transform: translateY(-3px); box-shadow: 3px 3px 8px rgba(0,0,0,0.15); }
[data-testid="stMetricValue"] { color: #004B87 !important; font-weight: 800; }
div.stButton > button:first-child { background: linear-gradient(90deg, #004B87, #0066cc); color: #FFFFFF; border: none; border-radius: 6px; font-weight: bold; padding: 10px 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
div.stButton > button:first-child:hover { background: linear-gradient(90deg, #E31837, #ff3333); transform: scale(1.02); }
[data-testid="stSidebar"] { border-right: 3px solid #E31837; }
[data-testid="stSidebar"]::before { content: "REMOVE BEFORE FLIGHT"; display: block; background-color: #E31837; color: white; font-family: monospace; font-weight: bold; text-align: center; padding: 6px; letter-spacing: 1.5px; margin-bottom: 20px; border-radius: 0 0 5px 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: left; background-color: #E31837; color: white; display: inline-block; padding: 3px 12px; font-family: monospace; font-weight: bold; border-radius: 4px; font-size: 13px; box-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>by Fuat Arıkan</div>", unsafe_allow_html=True)

col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️ LIFT-UP: Kestirimci Bakım Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 15px; font-weight: bold; font-style: italic; letter-spacing: 1px;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

with col_logo:
    if os.path.exists("agtoe.png"): st.image("agtoe.png", width=150)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", width=150)

# --- 3. AKILLI TAKIM DEĞİŞİMİ VE FİLTRASYON MOTORU ---
def filtrele_ve_takim_degisimini_bul(raw_blocks, raw_vals):
    if not raw_vals: return [], []
    guncel_vals = [raw_vals[0]]
    guncel_blocks = [raw_blocks[0]]
    current_max = raw_vals[0]
    
    for i in range(1, len(raw_vals)):
        # 0.05 mm'den büyük ani düşüş "Takım Değişimidir"
        if raw_vals[i-1] - raw_vals[i] >= 0.05:
            guncel_vals = [raw_vals[i]]
            guncel_blocks = [raw_blocks[i]]
            current_max = raw_vals[i]
        else:
            if raw_vals[i] > current_max:
                current_max = raw_vals[i]
            guncel_vals.append(current_max)
            guncel_blocks.append(raw_blocks[i])
            
    return guncel_blocks, guncel_vals

# --- 4. GELİŞMİŞ PDF MOTORU (Dinamik Tolerans Korumalı & Virgül/Nokta Ayrımı) ---
def extract_pdf_data_advanced(file):
    rows_list = []
    son_okunan_ust_tolerans = 0.100 # Varsayılan güvenlik toleransı
    son_okunan_alt_tolerans = -0.100
    
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: continue
                for line in text.split('\n'):
                    line_clean = line.replace('"', '').strip()
                    if re.search(r'\d', line_clean) and not line_clean.startswith(("Name", "CMM", "Date", "Run", "Part")):
                        # Virgül veya noktayı standart noktaya çevirerek float dönüşümü
                        nums_str = re.findall(r'[-+]?\d+[.,]\d+|[-+]?\d+', line_clean.replace('mm', ''))
                        
                        if len(nums_str) >= 2:
                            ilk_sayi_index = line_clean.find(nums_str[0])
                            olcum_adi = line_clean[:ilk_sayi_index].strip()
                            if not olcum_adi: continue
                            
                            try:
                                nums = [float(n.replace(',', '.')) for n in nums_str]
                                meas = nums[0]
                                nom = nums[1] if len(nums) > 1 else 0.0
                                
                                # Eğer satırda 5 değer varsa (Toleranslar dosyada belirtilmişse)
                                if len(nums) >= 5:
                                    son_okunan_ust_tolerans = nums[2]
                                    son_okunan_alt_tolerans = nums[3]
                                    dev = nums[4]
                                # Eğer tolerans verilmemişse (Örn: 1_PROFILE.X) üstteki değeri miras al
                                elif len(nums) == 3:
                                    dev = nums[2]
                                else:
                                    dev = nums[-1]
                                    
                                rows_list.append({
                                    "Olcum_Adi": olcum_adi,
                                    "Olculen_Deger": meas,
                                    "Nominal": nom,
                                    "Sapma": abs(dev),
                                    "Ust_Tolerans": son_okunan_ust_tolerans, 
                                    "Alt_Tolerans": son_okunan_alt_tolerans
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

# --- 5. 3-FAZLI KESTİRİMCİ FİZİK MOTORU (Parabolik Şahlanma) ---
class AI_ToolLife:
    def __init__(self, birim_ad):
        self.birim_ad = birim_ad
        self.scenarios = {}

    def uclu_faz_modeli(self, blocks, wear_data, future_blocks, tolerance):
        x = np.array(blocks)
        y = np.array(wear_data)
        
        if len(x) < 2: return np.full(len(future_blocks), y[0] if len(y)>0 else 0)
            
        # 2. Faz: Çoğunluk Düz Gidiş (Lineer Eğim)
        eğim, kesisim = np.polyfit(x[-3:] if len(x) >= 3 else x, y[-3:] if len(x) >= 3 else y, 1)
        if eğim <= 0: eğim = 0.001 # Minimum aşınma ivmesi
        
        y_fut = []
        for bx in future_blocks:
            if bx <= x[-1]:
                val = np.interp(bx, x, y) # Gerçek veri
            else:
                lineer_tahmin = kesisim + eğim * bx
                # 3. Faz: Parabolik Kırılma Bölgesi
                uyari_siniri = tolerance * 0.60
                if lineer_tahmin > uyari_siniri:
                    # Toleransa yaklaştıkça x karesiyle artan parabolik büyüme
                    carpan = ((lineer_tahmin - uyari_siniri) / (tolerance - uyari_siniri))
                    faz3_siddeti = 0.15 * (carpan ** 2) * tolerance 
                    val = lineer_tahmin + faz3_siddeti
                else:
                    val = lineer_tahmin
            y_fut.append(val)
            
        return np.array(y_fut)

    def add_scenario(self, name, blocks, wear_data, tolerance, cam_cycle_time=None):
        if len(blocks) < 2:
            raise ValueError(f"{name} analizi için en az 2 ardışık parça ölçümü gerekiyor.")
            
        max_blok = max(blocks)
        veri_sayisi = len(blocks)
        
        future_blocks = np.arange(1, max_blok * 10 + 50)
        y_fut_array = self.uclu_faz_modeli(blocks, wear_data, future_blocks, tolerance)
        
        cross_idx = np.where(y_fut_array >= tolerance)[0]
        uzak_tahmin_uyarisi = False
        
        if len(cross_idx) > 0:
            exact_cross = float(future_blocks[cross_idx[0]])
            grafik_son_blok = min(int(exact_cross) + 3, 5000)
            if exact_cross > (max_blok * 5): uzak_tahmin_uyarisi = True
            
            guven_araligi_metni = f"{exact_cross:.1f} Adet Parça"
            uretim_metni = f"3-Fazlı modele göre aktif takımınız {exact_cross:.1f}. parçadan sonra maksimum tolerans sınırını aşacaktır."
            
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
            uretim_metni = "Analiz ufku boyunca risk gözlemlenmemiştir."
            
        self.scenarios[name] = {
            'b_raw': blocks, 'y_raw': wear_data, 'b_fut': future_blocks[:grafik_son_blok], 'y_fut': y_fut_array[:grafik_son_blok],
            'tolerance': tolerance, 'guven_araligi_metni': guven_araligi_metni, 
            'sure_araligi_metni': sure_araligi_metni, 'uretim_metni': uretim_metni, 
            'cam_cycle_time': cam_cycle_time if cam_cycle_time else 0.0,
            'veri_sayisi': veri_sayisi, 'uzak_tahmin_uyarisi': uzak_tahmin_uyarisi
        }

    def plot_single_scenario(self, name):
        data = self.scenarios[name]
        tol = data['tolerance']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
        uyari_siniri = tol * 0.70
        
        for ax, x_raw, x_fut, xlabel in zip(
            [ax1, ax2], 
            [data['b_raw'], [b * data['cam_cycle_time'] for b in data['b_raw']]],
            [data['b_fut'], [b * data['cam_cycle_time'] for b in data['b_fut']]],
            ["Ardışık İşlenen Parça Sırası (Adet)", "Toplam Aktif Kesme Süresi (Dakika)"]
        ):
            ax.axhspan(0, uyari_siniri, facecolor='#d4edda', alpha=0.4, label='Güvenli Bölge (Yeşil)')
            ax.axhspan(uyari_siniri, tol, facecolor='#fff3cd', alpha=0.5, label='Uyarı Bölgesi (Sarı)')
            ax.axhspan(tol, tol * 1.5, facecolor='#f8d7da', alpha=0.4, label='Risk Bölgesi (Kırmızı)')

            ax.axhline(tol, color='#dc3545', linewidth=3, linestyle='--', label=f"Dosya Toleransı ({tol} {self.birim_ad})")
            ax.axhline(uyari_siniri, color='#ffc107', linewidth=2.5, linestyle='--', label=f"Erken Uyarı Eşiği")

            ax.plot(x_fut, data['y_fut'], color='#004B87', linestyle='-', linewidth=4, zorder=4, label="3-Fazlı Aşınma Eğrisi")
            ax.scatter(x_raw, data['y_raw'], color='#E31837', s=160, zorder=5, edgecolor='white', linewidth=2, label='CMM Ölçümleri')
            
            ax.set_title(f"Kritik Bölge: {name.upper()}", fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Boyutsal Sapma [{self.birim_ad}]", fontsize=11, fontweight='bold')
            ax.set_ylim(0.0, tol * 1.3)
            if len(x_fut) > 0: ax.set_xlim(0, x_fut[-1] * 1.05)
            
            # Efsaneyi (Legend) grafiğin sağına, dışına taşıdık
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.9, borderaxespad=0.)
            ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

        fig.tight_layout(pad=3.0)
        return fig 

# --- 6. YAN MENÜ (SİDEBAR) ---
with st.sidebar:
    st.header(" ⚙️  Veri Giriş Sihirbazı")
    veri_giris_modu = st.radio("Sistemi Nasıl Kullanacaksınız?", ["CMM Dosyası Yükle (PDF/Otonom)", "Manuel Veri Girişi (Klasik)"])
    
    st.markdown("---")
    st.header(" 📏  Genel Ayarlar")
    birim_secimi = st.radio("Ölçüm Birimi Sistemi", ["Milimetre (mm)", "Mikron (µm)"], index=0, horizontal=True)
    is_mikron = "Mikron" in birim_secimi
    birim_ad = "Mikron" if is_mikron else "mm"
    
    senaryo_sayisi = st.number_input("İzlenecek Kritik Bölge Sayısı", min_value=1, max_value=5, value=1, step=1)
    
    if veri_giris_modu == "Manuel Veri Girişi (Klasik)":
        tol_ornek = "Örn: 5" if is_mikron else "Örn: 0.1"
        tol_siniri = st.number_input(f"Maksimum Tolerans Limiti ({birim_ad})", min_value=0.001, value=0.100, format="%g")

# --- 7. OTONOM DOSYA YÜKLEME ALGORİTMASI ---
df_ana = pd.DataFrame()
if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
    st.info("💡 **Otonom Mod:** PDF veya Excel formatındaki raporları sırayla sürükleyiniz. Tolerans değerleri doğrudan CMM dosyasından otomatik alınacaktır.")
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
                    secilen_olcum = st.selectbox("📏 Dosyadaki Karşılığı", olcumler, key=f"geo_{i}", index=None)
                    if not secilen_olcum: eksik_alanlar.append(f"{isim}: Bölge Seçimi")
                    aykiri_filtre = st.checkbox("Outlier Sapmaları Temizle", value=True, key=f"outlier_{i}")
                    cmm_str, bolge_toleransi = "", None
                    
                    # Seçilen bölge için Otonom Tolerans
                    if secilen_olcum:
                        df_sub = df_ana[df_ana['Olcum_Adi'] == secilen_olcum]
                        if 'Ust_Tolerans' in df_sub.columns and pd.notna(df_sub['Ust_Tolerans'].iloc[0]):
                            bolge_toleransi = df_sub['Ust_Tolerans'].iloc[0]
                else:
                    st.info("Eşleştirme listesi için geçerli PDF bekleniyor.")
                    eksik_alanlar.append(f"{isim}: CMM Dosyası")
                    secilen_olcum, aykiri_filtre, cmm_str, bolge_toleransi = None, False, "", None
            else:
                st.markdown("**Manuel Ölçüm Girişi**")
                cmm_str = st.text_input(f"Aşınma Değerleri ({birim_ad})", key=f"cmm_{i}")
                if not cmm_str: eksik_alanlar.append(f"{isim}: Manuel CMM Serisi")
                secilen_olcum, aykiri_filtre, bolge_toleransi = None, False, tol_siniri

        with colB:
            st.markdown("**CAM İşleme Çevrim Verisi**")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                cam_dk = st.number_input("Dakika", min_value=0, key=f"cam_dk_{i}", value=None)
                if cam_dk is None: eksik_alanlar.append(f"{isim}: İşleme Süresi")
            with t_col2:
                cam_sn = st.number_input("Saniye", min_value=0, max_value=59, key=f"cam_sn_{i}", value=None)
            
            cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None

        with colC:
            st.markdown("**🧠 Arka Plan Matematiği**")
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)" and secilen_olcum:
                st.success(f"✅ **{secilen_olcum}** bölgesi için {len(df_sub)} adet ardışık veri çekildi.")
                if bolge_toleransi:
                    st.info(f"📏 Bu bölgenin spesifik toleransı dosyadan **{bolge_toleransi} {birim_ad}** olarak ayarlandı.")
                else:
                    st.warning("⚠️ Dosyada Tolerans bulunamadı, manuel giriniz.")
                    bolge_toleransi = st.number_input(f"Manuel Tolerans ({birim_ad})", min_value=0.001, value=0.100, key=f"man_tol_{i}")

        senaryo_verileri.append({
            "isim": isim, "cmm_str": cmm_str, "secilen_olcum": secilen_olcum, 
            "aykiri_filtre": aykiri_filtre, "cam_sure": cam_sure, "bolge_toleransi": bolge_toleransi
        })

st.markdown("---")

if st.button(" 🚀  Kritik Bölge Kestirim Analizini Başlat", use_container_width=True, type="primary"):
    if len(eksik_alanlar) > 0:
        hata_metni = "\n".join([f"- {alan}" for alan in list(set(eksik_alanlar))])
        st.error(f" ⚠️  Eksik bilgileri doldurunuz:\n\n{hata_metni}")
        st.session_state.analiz_yapildi = False
    else:
        try:
            system = AI_ToolLife(birim_ad=birim_ad)
            for d in senaryo_verileri:
                aktif_tolerans = float(d["bolge_toleransi"])
                
                if veri_giris_modu == "Manuel Veri Girişi (Klasik)":
                    raw_vals = [float(x.replace(',', '.')) for x in d["cmm_str"].split()]
                    blocks, cmm_vals = filtrele_ve_takim_degisimini_bul(list(range(1, len(raw_vals) + 1)), raw_vals)
                else:
                    df_sub = df_ana[df_ana['Olcum_Adi'] == d["secilen_olcum"]].copy()
                    
                    if d["aykiri_filtre"]:
                        mean_val = df_sub['Sapma'].mean()
                        std_val = df_sub['Sapma'].std()
                        if std_val > 0:
                            df_sub = df_sub[np.abs((df_sub['Sapma'] - mean_val) / std_val) < 3]
                            
                    raw_vals = df_sub['Sapma'].tolist()
                    raw_blocks = df_sub['Parca_Sira'].tolist() if 'Parca_Sira' in df_sub.columns else list(range(1, len(raw_vals) + 1))
                    
                    # Takım Değişimi Algılaması (0.05 Eşiği)
                    blocks, cmm_vals = filtrele_ve_takim_degisimini_bul(raw_blocks, raw_vals)

                system.add_scenario(d["isim"], blocks, cmm_vals, tolerance=aktif_tolerans, cam_cycle_time=d["cam_sure"])
            
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
            st.info(f" 🎯  {veri['uretim_metni']}")

    st.markdown("---")
    st.subheader(" 📦 Analiz Paketini Kaydet")
    zip_isim = st.text_input("Rapor Klasörü Adı:", value="TOMTAS_LIFTUP_Rapor").strip() + ".zip"
    
    rapor_verileri = []
    for isim, veri in system.scenarios.items():
        ts = round(veri['cam_cycle_time'] * 60)
        sure_m = f"{ts // 60} Dk {ts % 60} Sn" if veri['cam_cycle_time'] > 0 else "Manuel Mod"

        rapor_verileri.append({
            "Kritik Bölge": isim, 
            "Çevrim Süresi": sure_m,
            "CMM Takım Aşınma Serisi": " - ".join([f"{x:.4f}" for x in veri['y_raw']]), 
            "Kırılma Ufku (Parça)": veri['guven_araligi_metni'], 
            "Kırılma Ufku (Zaman)": veri['sure_araligi_metni'],
            "Aşınma Özeti": veri['uretim_metni']
        })
    
    df_rapor = pd.DataFrame(rapor_verileri)
    
    excel_buffer, zip_buffer = io.BytesIO(), io.BytesIO()
    df_rapor.set_index("Kritik Bölge").T.reset_index().to_excel(excel_buffer, index=False)
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Kiyaslama_Matrisi.xlsx", excel_buffer.getvalue())
        for g_isim, f in cizilen_grafikler.items():
            img_buf = io.BytesIO()
            f.savefig(img_buf, format="png", bbox_inches="tight", dpi=300) 
            zf.writestr(f"{g_isim}_Grafik.png", img_buf.getvalue())

    st.download_button("📥 Paketi İndir (Excel + Sektörel Grafikler)", data=zip_buffer.getvalue(), file_name=zip_isim, mime="application/zip", use_container_width=True)
