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

# --- 1. SAYFA VE SABİT TANIMLAMALARI (NameError Önlemi) ---
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

# --- 2. AKILLI VERİ DÜZELTME FİLTRESİ ---
def veriyi_duzelt_rolling_max(wear_verileri):
    """CMM ölçümlerindeki prob zikzaklarını giderip kümülatif aşınma trendi (hep yukarı) üretir."""
    if not wear_verileri: return []
    wear = np.array(wear_verileri, dtype=float)
    islenmis = []
    current_max = wear[0]
    for w in wear:
        if w > current_max: current_max = w
        islenmis.append(current_max)
    return islenmis

# --- 3. ZEISS CALYPSO GELİŞMİŞ PDF MOTORU ---
def extract_pdf_data_advanced(file):
    """PDF içindeki karmaşık metin yapılarından isim ve sapma verilerini ayırt eder."""
    rows_list = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: continue
                
                for line in text.split('\n'):
                    line_clean = line.replace('"', '').strip()
                    
                    # Satırda en az bir sayı varsa ve tipik başlıklar değilse analiz et
                    if re.search(r'\d', line_clean) and not line_clean.startswith(("Name", "CMM", "Date", "Run", "Part")):
                        # Tüm sayısal blokları (noktalı/virgüllü) ayıkla
                        nums = re.findall(r'[-+]?\d+[\.,]\d+', line_clean)
                        
                        if len(nums) >= 2:
                            # Ölçüm adı genellikle ilk sayıdan önceki kısımdır
                            ilk_sayi_index = line_clean.find(nums[0])
                            olcum_adi = line_clean[:ilk_sayi_index].strip()
                            
                            try:
                                meas = float(nums[0].replace(',', '.'))
                                dev = float(nums[-1].replace(',', '.'))
                                nom = float(nums[1].replace(',', '.')) if len(nums) > 1 else 0.0
                                
                                if olcum_adi:
                                    rows_list.append({
                                        "Olcum_Adi": olcum_adi,
                                        "Olculen_Deger": meas,
                                        "Nominal": nom,
                                        "Sapma": abs(dev),
                                        "Ust_Tolerans": 0.100, 
                                        "Alt_Tolerans": -0.100
                                    })
                            except ValueError:
                                continue
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

# --- 4. KESTİRİMCİ FİZİK VE GRAFİK MOTORU ---
class AI_ToolLife:
    def __init__(self, tolerance, birim_ad):
        self.tolerance = float(tolerance)
        self.birim_ad = birim_ad
        self.scenarios = {}

    def add_scenario(self, name, blocks, wear_data, cam_cycle_time=None, t_theo=None, mat_name=None):
        if len(blocks) < 2:
            raise ValueError(f"{name} analizi için en az 2 ardışık parça ölçümü gerekiyor.")
            
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
        
        uzak_tahmin_uyarisi = False
        karsilastirma_durumu = "normal"
        
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
                sure_araligi_metni = f"{dk} Dk {sn} Sn"
                if t_theo:
                    if exact_time_minutes / t_theo >= 1.0: karsilastirma_durumu = "hata_buyuk"
                    elif exact_time_minutes / t_theo >= 0.75: karsilastirma_durumu = "tebrikler"
                    elif exact_time_minutes / t_theo <= 0.15: karsilastirma_durumu = "hata_kucuk"
            else:
                sure_araligi_metni = "Süre Belirtilmedi"
        else:
            grafik_son_blok = int(max_blok * 1.5)
            guven_araligi_metni = f"{grafik_son_blok}+ Parça"
            sure_araligi_metni = "Risk Sınırı Dışında"
            uretim_metni = "Analiz ufku boyunca bu bölgede herhangi bir boyutsal risk gözlemlenmemiştir."
            
        future_blocks = np.arange(1, grafik_son_blok + 1).reshape(-1, 1)
        future_y = model.predict(poly.transform(future_blocks))
        
        self.scenarios[name] = {
            'b_raw': blocks, 'y_raw': wear_data, 'b_fut': future_blocks.flatten(), 'y_fut': future_y,
            'rmse_val': rmse_val, 't_theo': t_theo if t_theo else 0.0,
            'guven_araligi_metni': guven_araligi_metni, 'sure_araligi_metni': sure_araligi_metni,
            'uretim_metni': uretim_metni, 'cam_cycle_time': cam_cycle_time if cam_cycle_time else 0.0,
            'veri_sayisi': veri_sayisi, 'uzak_tahmin_uyarisi': uzak_tahmin_uyarisi,
            'karsilastirma_durumu': karsilastirma_durumu, 'mat_name': mat_name if mat_name else "Otonom İşleme"
        }

    def plot_single_scenario(self, name):
        data = self.scenarios[name]
        fig, ax = plt.subplots(figsize=(10, 5))
        uyari_siniri = self.tolerance * 0.75
        
        # Üç Renkli Bölge Şeritleri
        ax.axhspan(0, uyari_siniri, facecolor='#d4edda', alpha=0.6, label='Güvenli İşleme Alanı (Yeşil)')
        ax.axhspan(uyari_siniri, self.tolerance, facecolor='#fff3cd', alpha=0.7, label='Erken Uyarı Alanı (Sarı)')
        ax.axhspan(self.tolerance, self.tolerance * 1.5, facecolor='#f8d7da', alpha=0.6, label='Boyutsal Risk Alanı (Kırmızı)')

        ax.axhline(self.tolerance, color='#dc3545', linewidth=3, linestyle='--', label=f"Maksimum Tolerans Limiti ({self.tolerance} {self.birim_ad})")
        ax.axhline(uyari_siniri, color='#ffc107', linewidth=2.5, linestyle='--', label=f"Erken Uyarı Sınırı ({uyari_siniri} {self.birim_ad})")

        ax.plot(data['b_fut'], data['y_fut'], color='#004B87', linestyle='-', linewidth=3.5, zorder=4, label="Aşınma Tahmin Eğrisi")
        ax.scatter(data['b_raw'], data['y_raw'], color='#E31837', s=140, zorder=5, edgecolor='white', linewidth=1.5, label='Kümülatif CMM Sapma Noktaları')
        
        ax.set_title(f"Kritik Bölge Kestirim Dashboard: {name.upper()}", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Ardışık Üretilen Parça Sırası (Adet)", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"CMM Boyutsal Ölçüm Sapması [{self.birim_ad}]", fontsize=10, fontweight='bold')
        ax.set_ylim(0.0, self.tolerance * 1.4)
        ax.set_xlim(0.5, data['b_fut'][-1])
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)
        fig.tight_layout()
        return fig 

# --- 5. ARAYÜZ VE CSS ---
st.markdown("""
<style>
header[data-testid="stHeader"] { background: linear-gradient(90deg, #004B87, #E31837) !important; height: 4px !important; }
h1, h2, h3, h4 { color: #004B87 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
[data-testid="metric-container"] { border: 1px solid rgba(136, 136, 136, 0.2); padding: 15px; border-radius: 8px; background-color: rgba(248, 249, 250, 0.05); box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
div.stButton > button:first-child { background: linear-gradient(90deg, #004B87, #0066cc); color: #FFFFFF; font-weight: bold; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
div.stButton > button:first-child:hover { background: linear-gradient(90deg, #E31837, #ff3333); transform: scale(1.02); }
[data-testid="stSidebar"] { border-right: 3px solid #E31837; }
</style>
""", unsafe_allow_html=True)

col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️ LIFT-UP: Kestirimci Bakım Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-style: italic;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

# --- 6. SİDEBAR ---
with st.sidebar:
    st.header(" ⚙️ Veri Giriş Sihirbazı")
    veri_giris_modu = st.radio("Sistemi Nasıl Kullanacaksınız?", ["CMM Dosyası Yükle (PDF/Otonom)", "Manuel Veri Girişi (Klasik)"])
    st.markdown("---")
    birim_secimi = st.radio("Ölçüm Birimi", ["Mikron (µm)", "Milimetre (mm)"], horizontal=True)
    birim_ad = "Mikron" if "Mikron" in birim_secimi else "mm"
    
    tol_ornek = "Örn: 5" if birim_ad == "Mikron" else "Örn: 0.05"
    tol_siniri = st.number_input(f"Maksimum Tolerans Limiti ({birim_ad})", value=None, format="%g", placeholder=tol_ornek)
    senaryo_sayisi = st.number_input("İzlenecek Kritik Bölge Sayısı", min_value=1, max_value=5, value=1)
    
    # Sadece Manuel Moddaysa Taylor Ek Parametreleri Görünsün
    if veri_giris_modu == "Manuel Veri Girişi (Klasik)":
        st.markdown("---")
        st.markdown("📜 **Klasik Mod Ayarları**")
        ortak_malzeme = st.checkbox("Tüm bölgelerde ortak malzeme kullan", value=True)
        genel_m_secim = st.selectbox("Ortak Hammadde", list(MALZEMELER.keys())) if ortak_malzeme else None

# --- 7. VERİ YÜKLEME ---
df_ana = pd.DataFrame()
if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
    st.info("💡 **Otonom Mod:** PDF veya Excel formatındaki CMM raporlarını sırayla sürükleyiniz. Taylor formülü ve takım verileri gizlenmiştir.")
    yuklenen_dosyalar = st.file_uploader("CMM Raporlarını Klasör Halinde Sürükleyin", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if yuklenen_dosyalar:
        veri_listesi = []
        for i, dosya in enumerate(yuklenen_dosyalar):
            df_t = extract_pdf_data_advanced(dosya) if dosya.name.lower().endswith('.pdf') else clean_cmm_data(pd.read_excel(dosya))
            if not df_t.empty:
                df_t['Parca_Sira'] = i + 1
                veri_listesi.append(df_t)
        
        if veri_listesi:
            df_ana = pd.concat(veri_listesi, ignore_index=True)
            if 'Olcum_Adi' in df_ana.columns: df_ana = df_ana.sort_values(by=['Olcum_Adi', 'Parca_Sira'])

# --- 8. DİNAMİK SEKMELER VE EŞLEŞTİRME ---
st.markdown(f"### 📋 Kritik İzleme Bölgeleri ({senaryo_sayisi} Bölge)")
sekmeler = st.tabs([f"{i+1}. Kritik Bölge" for i in range(senaryo_sayisi)])
senaryo_verileri = []
eksik_alanlar = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim = st.text_input("İzlenecek Bölge / Unsur İsmi", value=f"Kritik Bölge {i+1}", key=f"isim_{i}")
        colA, colB = st.columns([2, 1])

        with colA:
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)":
                if not df_ana.empty and 'Olcum_Adi' in df_ana.columns:
                    secilen_olcum = st.selectbox("📏 CMM Raporundan Eşleşen Bölgeyi Seçin", df_ana['Olcum_Adi'].unique(), key=f"geo_{i}", index=None)
                    if not secilen_olcum: eksik_alanlar.append(f"{isim}: Bölge Seçimi")
                    aykiri_filtre = st.checkbox("Hatalı (Outlier) Sapmaları Temizle", value=True, key=f"out_{i}")
                    cmm_str, cam_sure, t_theo, m_secim = "", None, None, None
                else:
                    st.warning("Veri çekilebilmesi için CMM raporu bekleniyor.")
                    eksik_alanlar.append(f"{isim}: Dosya Eksik")
                    secilen_olcum, aykiri_filtre, cmm_str, cam_sure, t_theo, m_secim = None, False, "", None, None, None
            else:
                st.markdown("**Manuel Kesme ve Taylor Parametreleri**")
                c1, c2 = st.columns(2)
                with c1:
                    m_secim = genel_m_secim if ortak_malzeme else st.selectbox("Malzeme", list(MALZEMELER.keys()), key=f"mat_{i}")
                    vc = st.number_input("Vc (m/min)", min_value=1, value=400, key=f"vc_{i}")
                    ae = st.number_input("ae (mm)", min_value=0.1, value=5.0, key=f"ae_{i}")
                    fz = st.number_input("fz (mm/diş)", min_value=0.01, value=0.08, key=f"fz_{i}")
                with c2:
                    t_cap = st.number_input("Takım Çapı (D)", min_value=1, value=6, key=f"d_{i}")
                    t_dis = st.number_input("Diş Sayısı (z)", min_value=1, value=4, key=f"z_{i}")
                    cam_dk = st.number_input("Çevrim (Dk)", min_value=0, value=2, key=f"dk_{i}")
                
                cam_sure = cam_dk
                cmm_str = st.text_input(f"Aşınma Verileri ({birim_ad})", placeholder="Örn: 0.01 0.04 0.08", key=f"cmm_{i}")
                
                if m_secim and t_cap and vc and ae:
                    daf = 1.8 if ae >= t_cap else 1.4
                    t_theo = (MALZEMELER[m_secim]['c_taylor'] / (vc**3.5)) * (1 / (daf**1.5))
                else: t_theo = None
                
                if not cmm_str: eksik_alanlar.append(f"{isim}: Manuel Veri")
                secilen_olcum, aykiri_filtre = None, False

        with colB:
            st.markdown("**🧠 Kestirim Paneli**")
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)" and secilen_olcum:
                veri_adeti = len(df_ana[df_ana['Olcum_Adi'] == secilen_olcum])
                st.success(f"✅ {veri_adeti} adet parça verisi eşleştirildi.")
                st.info("Taylor Takım Formülü Otonom Modda Devre Dışı Bırakılmıştır. Sadece Bölge Aşınma Trendi İzlenir.")

        senaryo_verileri.append({
            "isim": isim, "cmm_str": cmm_str, "secilen_olcum": secilen_olcum, "aykiri_filtre": aykiri_filtre,
            "cam_sure": cam_sure, "t_theo": t_theo, "mat_isim": m_secim if m_secim else "Otonom"
        })

st.markdown("---")

# --- 9. ANALİZ MOTORU ---
if st.button(" 🚀 Kritik Bölge Kestirim Analizini Başlat", use_container_width=True, type="primary"):
    if tol_siniri is None: eksik_alanlar.append("Genel Ayarlar: Tolerans Limiti")
    
    if len(eksik_alanlar) > 0:
        st.error("⚠️ Eksik bilgileri doldurunuz:\n" + "\n".join([f"- {a}" for a in set(eksik_alanlar)]))
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
                    if d["aykiri_filtre"]:
                        mean_v, std_v = df_sub['Sapma'].mean(), df_sub['Sapma'].std()
                        if std_v > 0: df_sub = df_sub[np.abs((df_sub['Sapma'] - mean_v) / std_v) < 3]
                            
                    raw_vals = df_sub['Sapma'].tolist()
                    cmm_vals = veriyi_duzelt_rolling_max(raw_vals)
                    blocks = list(range(1, len(cmm_vals) + 1))

                system.add_scenario(d["isim"], blocks, cmm_vals, d["cam_sure"], d["t_theo"], d["mat_isim"])
            
            st.session_state.sistem_verisi = system
            st.session_state.analiz_yapildi = True

        except Exception as e:
            st.error(f"Modelleme hatası: Yeterli veri yok. ({e})")
            st.session_state.analiz_yapildi = False

# --- 10. SONUÇLAR VE İNDİRME ---
if st.session_state.analiz_yapildi and st.session_state.sistem_verisi:
    system = st.session_state.sistem_verisi
    
    st.markdown("### 📊 Bölge Bazlı Kestirim Sonuçları")
    sonuc_sekmeleri = st.tabs(list(system.scenarios.keys()))
    cizilen_grafikler = {}
    
    for idx, (isim, veri) in enumerate(system.scenarios.items()):
        with sonuc_sekmeleri[idx]:
            fig = system.plot_single_scenario(isim)
            st.pyplot(fig)
            cizilen_grafikler[isim] = fig 
            
            c1, c2 = st.columns(2)
            c1.metric("Kestirim Ömür Sınırı", veri['guven_araligi_metni'])
            if veri['t_theo'] > 0: c2.metric("Taylor Teorik Sınır", f"{veri['t_theo']:.1f} Dk")

            st.info(f" 🎯 {veri['uretim_metni']}")
            if veri['veri_sayisi'] < 3: st.error("⚠️ Kararlı tahmin için en az 3 parça verisi gerekir.")

    st.markdown("---")
    st.subheader(" 📦 Analiz Raporunu Kaydet")
    zip_isim = st.text_input("Rapor Klasörü Adı:", value="LIFTUP_Rapor").strip() + ".zip"
    
    rapor_verileri = []
    for isim, v in system.scenarios.items():
        rapor_verileri.append({
            "Kritik Bölge": isim, 
            "CMM Aşınma Serisi": " - ".join([f"{x:.4f}" for x in v['y_raw']]), 
            "Tahmini Kırılma Ufku": v['guven_araligi_metni'], 
            "Model Sapması (RMSE)": round(v['rmse_val'], 4)
        })
    
    excel_buffer, zip_buffer = io.BytesIO(), io.BytesIO()
    pd.DataFrame(rapor_verileri).set_index("Kritik Bölge").T.reset_index().to_excel(excel_buffer, index=False)
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Kiyaslama_Matrisi.xlsx", excel_buffer.getvalue())
        for g_isim, f in cizilen_grafikler.items():
            img_buf = io.BytesIO()
            f.savefig(img_buf, format="png", bbox_inches="tight", dpi=300) 
            zf.writestr(f"{g_isim}_Grafik.png", img_buf.getvalue())

    st.download_button("📥 Paketi İndir (Excel + Grafikler)", data=zip_buffer.getvalue(), file_name=zip_isim, mime="application/zip", use_container_width=True)
