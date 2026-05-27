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

st.set_page_config(page_title="LIFT-UP Kestirimci Bakım", page_icon=" ✈️ ", layout="wide")

# --- KALICI HAFIZA (SESSION STATE) TANIMLAMALARI ---
if 'ilk_giris' not in st.session_state:
    st.session_state.ilk_giris = True
if 'analiz_yapildi' not in st.session_state:
    st.session_state.analiz_yapildi = False
if 'sistem_verisi' not in st.session_state:
    st.session_state.sistem_verisi = None

# --- KARŞILAMA EKRANI (POP-UP) MANTIĞI ---
@st.dialog(" ✈️  LIFT-UP Sistemine Hoş Geldiniz")
def rehber_dialog():
    st.markdown("""
    **Bu sistem, İstatistiksel Regresyon ve Taylor Denklemlerini harmanlayarak takım ömrünü kestirimci olarak tahmin eder.**

    ###  🛠️  Nasıl Kullanılır?
    1. **Veri Giriş Yöntemi:** Sol menüden CMM dosya yükleme veya manuel giriş yöntemini seçin.
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

# --- HEADER ---
col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️  LIFT-UP: Kestirimci Bakım Sistemi</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 15px; font-weight: bold; font-style: italic; letter-spacing: 1px;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

with col_logo:
    if os.path.exists("agtoe.png"): st.image("agtoe.png", width=150)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", width=150)

# --- ANA MOTOR SINIFI ---
class AI_ToolLife:
    def __init__(self, tolerance, birim_ad):
        self.tolerance = tolerance
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

    def plot_dashboard(self):
        if not self.scenarios: return None
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        colors = ['#E31837', '#004B87', '#d62728', '#1f77b4', '#ff7f0e']

        genel_max_blok = max([data['b_fut'][-1] for data in self.scenarios.values()])
        genel_max_time = max([data['b_fut'][-1] * data['cam_cycle_time'] for data in self.scenarios.values()])
        
        for i, (name, data) in enumerate(self.scenarios.items()):
            col = colors[i % len(colors)]
            st.markdown(f"<div style='background-color:{col}; color:white; padding:5px 15px; border-radius:5px; display:inline-block; margin-top:15px; font-weight:bold; box-shadow: 2px 2px 4px rgba(0,0,0,0.2);'> 📌  {name.upper()} | Alaşım: {data['mat_name']}</div>", unsafe_allow_html=True)

            ax1.scatter(data['b_raw'], data['y_raw'], color=col, s=100, zorder=3, edgecolor='white', linewidth=1)
            ax1.plot(data['b_fut'], data['y_fut'], color=col, linestyle='--', linewidth=3, label=f"{name}", zorder=2)
            guven_bandi = data['rmse_val'] * 2
            ax1.fill_between(data['b_fut'], data['y_fut'] - guven_bandi, data['y_fut'] + guven_bandi, color=col, alpha=0.12)
            
            time_raw = [b * data['cam_cycle_time'] for b in data['b_raw']]
            time_fut = [b * data['cam_cycle_time'] for b in data['b_fut']]
            ax2.scatter(time_raw, data['y_raw'], color=col, s=100, zorder=3, edgecolor='white', linewidth=1)
            ax2.plot(time_fut, data['y_fut'], color=col, linestyle='-', linewidth=3, label=f"{name}", zorder=2)
            ax2.fill_between(time_fut, np.array(data['y_fut']) - guven_bandi, np.array(data['y_fut']) + guven_bandi, color=col, alpha=0.12)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Teorik Takım Ömrü", f"{data['t_theo']:.1f} Dakika")
            col2.metric("Tahmini Kırılma Ufku (Blok)", data['guven_araligi_metni'])
            col3.metric("Tahmini Kırılma Ufku (Zaman)", data['sure_araligi_metni'])

            st.info(f" 🎯  **Tahmini Aşınma Noktası:** {data['uretim_metni']}")

            if data['veri_sayisi'] < 3: st.error(" ⚠️  **Düşük Veri Yoğunluğu:** Kestirimci modele 3'ten az ölçüm girilmiştir.")
            if data['uzak_tahmin_uyarisi']: st.warning(" 🔭  **Aşırı Uzak Tahmin:** Uzun vadeli tahminler istatistiksel olarak yanıltıcı olabilir.")
            if data['karsilastirma_durumu'] == "hata_buyuk": st.error(f" 🛑  **Fiziksel Tutarsızlık İhtimali:** Hesaplanan süre Teorik Takım Ömrünü ({data['t_theo']:.1f} Dk) aşıyor.")
            elif data['karsilastirma_durumu'] == "tebrikler": st.success(f" 🏆  **Mükemmel Optimizasyon:** Takım ömrünüz teorik fiziksel sınırlara çok yakın!")
            elif data['karsilastirma_durumu'] == "hata_kucuk": st.error(f" ⚠️  **Aşırı Erken Aşınma:** Takım ömrü teorik değerin %15'inden bile daha az!")
            
            rmse_sinir = 10.0 if self.birim_ad == "Mikron" else 0.01
            if data['rmse_val'] > rmse_sinir: st.warning(f" ⚠️  **Veri Anomalyası:** Sapma: {data['rmse_val']:.4f} {self.birim_ad}.")
            st.divider()

        y_limit = self.tolerance * 1.5
        for ax, title, xlabel in zip([ax1, ax2], ["Blok Sayısına Göre Takım Aşınması", "CAM Süresine Göre Takım Aşınması"], ["İşlenen Sütun / Blok Sayısı", "Aktif CAM İşleme Süresi (Dakika)"]):
            ax.axhline(y=self.tolerance, color='#ff0000', linewidth=4, label=f"Tolerans ({self.tolerance} {self.birim_ad})", zorder=1)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            ax.set_ylabel(f"Boyutsal Sapma ({self.birim_ad})", fontsize=12, fontweight='bold')
            ax.set_ylim(0.0, y_limit)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

        ax1.set_xlim(0, genel_max_blok)
        ax2.set_xlim(0, genel_max_time)
        fig.tight_layout(pad=2.0)
        st.pyplot(fig)
        return fig 

MALZEMELER = {
    "Alüminyum 6061-T6": {"kc": 800, "c_taylor": 4.5e10}, "Alüminyum 7075-T6": {"kc": 975, "c_taylor": 3.5e10},
    "Titanyum Ti-6Al-4V": {"kc": 2100, "c_taylor": 1.2e10}, "Paslanmaz Çelik 304": {"kc": 2100, "c_taylor": 2.8e10},
    "17-4 PH Paslanmaz": {"kc": 2600, "c_taylor": 2.0e10}, "AISI 4340 Alaşımlı Çelik": {"kc": 2700, "c_taylor": 1.8e10}
}

# --- ADIM 1: SİHİRBAZ VE VERİ GİRİŞ SEÇİMİ ---
with st.sidebar:
    st.header(" ⚙️  Veri Giriş Sihirbazı")
    veri_giris_modu = st.radio("Sistemi Nasıl Kullanacaksınız?", ["CMM Dosyası Yükle (Otonom)", "Manuel Veri Girişi (Klasik)"])
    
    st.markdown("---")
    st.header(" 📏  Genel Ayarlar")
    birim_secimi = st.radio("Ölçüm Birimi Sistemi", ["Mikron (µm)", "Milimetre (mm)"], horizontal=True)
    is_mikron = "Mikron" in birim_secimi
    birim_ad = "Mikron" if is_mikron else "mm"
    
    tol_siniri = st.number_input(f"Maksimum Tolerans ({birim_ad})", value=None, format="%g")
    senaryo_sayisi = st.number_input("Karşılaştırılacak Takım/Senaryo Sayısı", min_value=1, max_value=5, value=1, step=1)
    
    st.markdown("---")
    ortak_malzeme = st.checkbox("Tüm senaryolarda ortak MALZEME kullan", value=True)
    genel_malzeme_secimi = st.selectbox("Ortak Hammadde Seçimi", list(MALZEMELER.keys()), index=None) if ortak_malzeme else None
    
    ortak_takim = st.checkbox("Tüm senaryolarda ortak TAKIM kullan", value=True)
    genel_t_cap = st.number_input("Ortak Takım Çapı (D) [mm]", value=None, min_value=1) if ortak_takim else None
    genel_t_dis = st.number_input("Ortak Takım Diş Sayısı (z)", value=None, min_value=1) if ortak_takim else None
    genel_t_boy = st.number_input("Ortak Takım Kesme Boyu (Lc) [mm]", value=None, min_value=1) if ortak_takim else None

# --- ADIM 2: DOSYA YÜKLEME VE PANDAS İŞLEMLERİ ---
df_ana = pd.DataFrame()
if veri_giris_modu == "CMM Dosyası Yükle (Otonom)":
    st.info("💡 **Otonom Mod:** CMM cihazınızdan aldığınız çoklu parçalara ait raporları (.xlsx veya .csv) doğrudan sisteme sürükleyin.")
    
    # Şablon İndirme Butonu
    df_sablon = pd.DataFrame({
        "Parca_Sira": [1, 2, 3], "Olcum_Adi": ["1_Ø14.5mm", "1_Ø14.5mm", "1_Ø14.5mm"],
        "Nominal": [14.5, 14.5, 14.5], "Ust_Tolerans": [0.1, 0.1, 0.1], 
        "Alt_Tolerans": [0.1, 0.1, 0.1], "Olculen_Deger": [14.51, 14.53, 14.56]
    })
    sablon_csv = df_sablon.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📄 Örnek CMM Veri Şablonunu İndir", data=sablon_csv, file_name="Ornek_CMM_Sablonu.csv", mime="text/csv")

    yuklenen_dosyalar = st.file_uploader("CMM Verilerini Yükleyin (Tek veya Çoklu Dosya)", type=['csv', 'xlsx'], accept_multiple_files=True)
    
    if yuklenen_dosyalar:
        veri_listesi = []
        for dosya in yuklenen_dosyalar:
            try:
                # Virgül/nokta problemini Pandas aşamasında çözüyoruz
                if dosya.name.endswith('.csv'): df_temp = pd.read_csv(dosya, sep=None, engine='python', decimal='.')
                else: df_temp = pd.read_excel(dosya)
                
                if 'Olculen_Deger' in df_temp.columns and df_temp['Olculen_Deger'].dtype == object:
                    df_temp['Olculen_Deger'] = df_temp['Olculen_Deger'].str.replace(',', '.').astype(float)
                    
                veri_listesi.append(df_temp)
            except Exception as e:
                st.error(f"Dosya okuma hatası ({dosya.name}): {e}")
                
        if veri_listesi:
            df_ana = pd.concat(veri_listesi, ignore_index=True)
            # Eğer parçalar karışıksa sıraya diz
            if 'Parca_Sira' in df_ana.columns: df_ana = df_ana.sort_values(by=['Olcum_Adi', 'Parca_Sira'])

st.markdown(f"###  📋  Parametre Girişi ve Eşleştirme ({senaryo_sayisi} Takım/Senaryo)")
sekmeler = st.tabs([f"{i+1}. Senaryo" for i in range(senaryo_sayisi)])
senaryo_verileri = []
eksik_alanlar = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim = st.text_input(f"Senaryo / Takım Adı", value=f"Takım {i+1}", key=f"isim_{i}")
        colA, colB, colC = st.columns([1.3, 1.3, 1])

        with colA:
            st.markdown("**Malzeme ve Takım Ayarları**")
            m_secim = genel_malzeme_secimi if ortak_malzeme else st.selectbox("Hammadde Seçimi", list(MALZEMELER.keys()), key=f"mat_{i}")
            s_malzeme = MALZEMELER.get(m_secim)
            if not s_malzeme: eksik_alanlar.append(f"{isim}: Hammadde")

            t_cap = genel_t_cap if ortak_takim else st.number_input("Takım Çapı (D)", min_value=1, key=f"tcap_{i}")
            t_dis = genel_t_dis if ortak_takim else st.number_input("Diş Sayısı (z)", min_value=1, key=f"tdis_{i}")
            t_boy = genel_t_boy if ortak_takim else st.number_input("Kesme Boyu (Lc)", min_value=1, key=f"tboy_{i}")
            if t_cap is None: eksik_alanlar.append(f"{isim}: Takım Çapı")

        with colB:
            st.markdown("**Kesme Parametreleri**")
            vc = st.number_input("Kesme Hızı (Vc) [m/min]", min_value=1, key=f"vc_{i}")
            fz = st.number_input("İlerleme (fz) [mm/diş]", format="%g", key=f"fz_{i}")
            ap = st.number_input("Eksenel Derinlik (ap)", format="%g", key=f"ap_{i}")
            ae = st.number_input("Radyal Derinlik (ae)", format="%g", key=f"ae_{i}")
            cam_dk = st.number_input("Dakika", min_value=0, key=f"cam_dk_{i}")
            cam_sn = st.number_input("Saniye", min_value=0, max_value=59, key=f"cam_sn_{i}")
            cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None
            
            if vc is None: eksik_alanlar.append(f"{isim}: Kesme Hızı")

            # --- OTONOM vs MANUEL CMM SEÇİMİ ---
            secilen_olcum = None
            aykiri_filtre = False
            cmm_str = ""
            if veri_giris_modu == "CMM Dosyası Yükle (Otonom)":
                if not df_ana.empty and 'Olcum_Adi' in df_ana.columns:
                    olcumler = df_ana['Olcum_Adi'].unique()
                    secilen_olcum = st.selectbox("📏 Analiz Edilecek Geometri (Dosyadan)", olcumler, key=f"geo_{i}")
                    aykiri_filtre = st.checkbox("Aykırı (Hatalı) Ölçümleri Filtrele", value=True, key=f"outlier_{i}")
                else:
                    st.warning("Geçerli bir CMM dosyası yüklenmedi.")
                    eksik_alanlar.append(f"{isim}: CMM Dosyası")
            else:
                cmm_str = st.text_input(f"CMM Verileri ({birim_ad}, Boşluklu)", key=f"cmm_{i}")
                if not cmm_str: eksik_alanlar.append(f"{isim}: CMM Verileri")

        with colC:
            st.markdown("** 🧠 CMM Sağlık Kontrolü & Motor**")
            if s_malzeme: st.info(f"**Kc:** {s_malzeme['kc']} MPa | **Taylor:** {s_malzeme['c_taylor']:.1e}")
            
            # --- SAĞLIK KONTROLÜ VE CPK HESAPLAMASI ---
            if veri_giris_modu == "CMM Dosyası Yükle (Otonom)" and secilen_olcum:
                df_sub = df_ana[df_ana['Olcum_Adi'] == secilen_olcum]
                if not df_sub.empty and 'Olculen_Deger' in df_sub.columns:
                    mean_val = df_sub['Olculen_Deger'].mean()
                    std_val = df_sub['Olculen_Deger'].std()
                    nom = df_sub['Nominal'].iloc[0]
                    usl = nom + df_sub['Ust_Tolerans'].iloc[0]
                    lsl = nom - df_sub['Alt_Tolerans'].iloc[0]
                    
                    # Cpk Hesaplama
                    if pd.notna(std_val) and std_val > 0:
                        cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val))
                    else: cpk = 0.0
                    
                    # Tolerans içi yüzde
                    in_spec = df_sub[(df_sub['Olculen_Deger'] <= usl) & (df_sub['Olculen_Deger'] >= lsl)]
                    in_spec_pct = len(in_spec) / len(df_sub) * 100
                    
                    st.markdown(f"""
                    <div style='background-color:rgba(248, 249, 250, 0.05); padding:10px; border-radius:5px; border-left: 3px solid #28a745; margin-bottom: 10px;'>
                    <b>📊 {secilen_olcum} Analizi:</b><br>
                    Tolerans İçi Oran: <b>%{in_spec_pct:.1f}</b><br>
                    Proses Yeterliliği (Cpk): <b>{cpk:.2f}</b>
                    </div>
                    """, unsafe_allow_html=True)

        # Veri setini oluşturup arka plana aktarma listesine ekliyoruz
        senaryo_verileri.append({
            "isim": isim, "mat_isim": m_secim, "mat_data": s_malzeme,
            "t_cap": t_cap, "t_dis": t_dis, "t_boy": t_boy,
            "vc": vc, "fz": fz, "ap": ap, "ae": ae, "cam_sure": cam_sure, 
            "cmm_str": cmm_str, "secilen_olcum": secilen_olcum, "aykiri_filtre": aykiri_filtre
        })

st.markdown("---")

if st.button(" 🚀  Takım Ömrü Tahminini Başlat", use_container_width=True, type="primary"):
    if tol_siniri is None: eksik_alanlar.append("Genel Ayarlar: Tolerans")
    
    if len(eksik_alanlar) > 0:
        hata_metni = "\n".join([f"- {alan}" for alan in list(set(eksik_alanlar))])
        st.error(f" ⚠️  Lütfen analizi başlatmadan önce aşağıdaki eksik bilgileri doldurunuz:\n\n{hata_metni}")
        st.session_state.analiz_yapildi = False
    else:
        try:
            system = AI_ToolLife(tolerance=tol_siniri, birim_ad=birim_ad)
            for d in senaryo_verileri:
                
                # Otonom mu Manuel mi kontrolü
                if veri_giris_modu == "Manuel Veri Girişi (Klasik)":
                    cmm_vals = [float(x.replace(',', '.')) for x in d["cmm_str"].split()]
                    blocks = list(range(1, len(cmm_vals) + 1))
                else:
                    # Pandas içinden o boyuta ait verileri çekiyoruz
                    df_sub = df_ana[df_ana['Olcum_Adi'] == d["secilen_olcum"]].copy()
                    
                    # Aykırı Değer Filtresi
                    if d["aykiri_filtre"]:
                        mean_val = df_sub['Olculen_Deger'].mean()
                        std_val = df_sub['Olculen_Deger'].std()
                        if std_val > 0:
                            z_scores = np.abs((df_sub['Olculen_Deger'] - mean_val) / std_val)
                            df_sub = df_sub[z_scores < 3]
                            
                    # Mutlak sapma (aşınma) hesabı: Abs(Ölçülen - Nominal)
                    cmm_vals = np.abs(df_sub['Olculen_Deger'] - df_sub['Nominal']).tolist()
                    blocks = df_sub['Parca_Sira'].tolist() if 'Parca_Sira' in df_sub.columns else list(range(1, len(cmm_vals) + 1))

                system.add_scenario(d["isim"], d["mat_isim"], d["mat_data"]['kc'], d["mat_data"]['c_taylor'], d["t_cap"], d["t_dis"], d["t_boy"], d["vc"], d["fz"], d["ap"], d["ae"], blocks, cmm_vals, d["cam_sure"])
            
            st.session_state.sistem_verisi = system
            st.session_state.analiz_yapildi = True

        except Exception as e:
            st.error(f"Sayısal format veya dosya işleme hatası: {e}.")
            st.session_state.analiz_yapildi = False

# --- RAPORLAMA ---
if st.session_state.analiz_yapildi and st.session_state.sistem_verisi is not None:
    system = st.session_state.sistem_verisi
    fig = system.plot_dashboard()

    st.markdown("---")
    st.subheader(" 📦 Tüm Analiz Paketini Kaydet")
    
    dosya_ismi_girdisi = st.text_input("📁 İndirilecek Dosyanın Adını Belirleyin:", value="LIFTUP_Rapor")
    temiz_isim = dosya_ismi_girdisi.strip() if dosya_ismi_girdisi.strip() else "LIFTUP_Rapor"
        
    zip_isim = f"{temiz_isim}.zip"
    excel_isim = f"{temiz_isim}_Detay.xlsx"
    
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
            "Hesaplanan Sapma Verileri (Wear)": " - ".join([f"{x:.4f}" for x in veri['y_raw']]), 
            "Teorik Takım Ömrü": temiz_t_theo_metni,
            "Tahmini Kırılma Ufku (Blok)": veri['guven_araligi_metni'],
            "Tahmini Kırılma Ufku (Zaman)": veri['sure_araligi_metni'],
            "Model Hata Sapması (RMSE)": round(veri['rmse_val'], 4),
            "Tahmini Aşınma Noktası": veri['uretim_metni'].replace('*', '') 
        })
    
    df_rapor = pd.DataFrame(rapor_verileri)
    df_rapor.set_index("Senaryo Adı", inplace=True)
    df_rapor_final = df_rapor.T.reset_index()
    df_rapor_final.rename(columns={'index': 'Parametreler ve Sonuçlar'}, inplace=True)
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_rapor_final.to_excel(writer, index=False, sheet_name='Kıyaslama Analizi')
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(excel_isim, excel_buffer.getvalue())
        if fig is not None:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300) 
            zip_file.writestr(f"{temiz_isim}_Grafikler.png", img_buffer.getvalue())

    st.download_button(
        label="📥 " + zip_isim + " Paketini İndir",
        data=zip_buffer.getvalue(),
        file_name=zip_isim,
        mime="application/zip",
        use_container_width=True
    )
