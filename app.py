import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="LIFT-UP Kestirimci Bakım", page_icon="✈️", layout="wide")

# --- KARŞILAMA EKRANI (POP-UP) MANTIĞI ---
@st.dialog("✈️ LIFT-UP Sistemine Hoş Geldiniz")
def rehber_dialog():
    st.markdown("""
    **Bu sistem, Makine Öğrenmesi ve Taylor Denklemlerini harmanlayarak takım ömrünü otonom olarak tahmin eder.**
    
    ### 🛠️ Nasıl Kullanılır?
    1. **Birim ve Tolerans:** Sol menüden ölçüm biriminizi (Mikron/mm) ve tezgâhın maksimum aşınma toleransını girin.
    2. **Parametreler:** Kullanacağınız malzeme, takım ölçüleri ve CAM kesme verilerini eksiksiz doldurun.
    3. **CMM Verileri:** Ölçtüğünüz aşınma değerlerini aralarında boşluk bırakarak yazın *(Örn: 0.2 0.5 0.8...)*.
    4. **Analiz:** 'Tahmini Başlat' butonuna basın ve yapay zekanın operasyon önerilerini inceleyin.
    
    *(Bu bilgilendirme penceresini sağ üstteki 'X' işaretine basarak kapatabilirsiniz.)*
    """)

if 'ilk_giris' not in st.session_state:
    st.session_state.ilk_giris = True

if st.session_state.ilk_giris:
    rehber_dialog()
    st.session_state.ilk_giris = False
# ----------------------------------------

# --- CSS İLE TEMA VE DÜZEN ENJEKSİYONU ---
st.markdown("""
<style>
    /* ÜST TAVAN ŞERİDİ (PREMIUM BANNER) - Yan menüyle hizalı */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #004B87, #E31837) !important;
        height: 4px !important;
    }

    /* ÜST BOŞLUK DÜZELTİLDİ - Yazıların kaybolmaması için padding 3rem yapıldı */
    .block-container {
        padding-top: 3rem !important; 
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Başlıklar */
    h1, h2, h3, h4 {
        color: #004B87 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }

    /* METRİK KARTLARI */
    [data-testid="metric-container"] {
        border: 1px solid rgba(136, 136, 136, 0.2);
        padding: 15px;
        border-radius: 8px;
        background-color: rgba(248, 249, 250, 0.05); 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 3px 3px 8px rgba(0,0,0,0.15);
    }
    
    [data-testid="stMetricValue"] {
        color: #004B87 !important;
        font-weight: 800;
    }

    /* Ana Buton (Başlat Tuşu) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #004B87, #0066cc);
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #E31837, #ff3333); 
        color: #FFFFFF;
        transform: scale(1.02); 
        box-shadow: 0 6px 10px rgba(0,0,0,0.25);
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #E31837 !important;
        color: #E31837 !important;
        font-weight: bold;
    }

    /* YAN PANEL (SIDEBAR) STİLİ */
    [data-testid="stSidebar"] {
        border-right: 3px solid #E31837;
    }

    [data-testid="stSidebar"]::before {
        content: "REMOVE BEFORE FLIGHT";
        display: block;
        background-color: #E31837;
        color: white;
        font-family: monospace;
        font-weight: bold;
        text-align: center;
        padding: 6px;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
        border-radius: 0 0 5px 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* SAĞ PANEL KAPSÜLÜ (Eksik olduğu için HTML bozuk görünüyordu, eklendi) */
    .right-panel-card {
        background-color: rgba(136, 136, 136, 0.06); 
        border-left: 4px solid #004B87;
        border-top: 4px solid #E31837;
        border-radius: 8px;
        padding: 20px;
        height: 100%;
        box-shadow: -3px 4px 15px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------

# --- İMZA YERİ ---
st.markdown("<div style='text-align: left; background-color: #E31837; color: white; display: inline-block; padding: 3px 12px; font-family: monospace; font-weight: bold; border-radius: 4px; font-size: 13px; box-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>by Fuat Arıkan</div>", unsafe_allow_html=True)

# --- HEADER ---
col_baslik, col_logo = st.columns([5, 1])

with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🛠️ LIFT-UP: Kestirimci Bakım Sistemi</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 15px; font-weight: bold; font-style: italic; letter-spacing: 1px;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)

with col_logo:
    if os.path.exists("agtoe.png"):
        st.image("agtoe.png", width=150)
    elif os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=150)

st.markdown("<br>", unsafe_allow_html=True)

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
            exact_cross = min(valid_roots) 
            grafik_son_blok = min(int(np.ceil(exact_cross)) + 2, 5000)
            
            if exact_cross > (max_blok * 5):
                uzak_tahmin_uyarisi = True

            tam_blok = int(exact_cross)
            yuzde = int(round((exact_cross - tam_blok) * 100))
            if yuzde == 100:
                tam_blok += 1
                yuzde = 0
                
            exact_time_minutes = exact_cross * cam_cycle_time
            dk = int(exact_time_minutes)
            sn = int(round((exact_time_minutes - dk) * 60))
            if sn == 60:
                dk += 1
                sn = 0

            oran = exact_time_minutes / t_theo
            if oran >= 1.0:
                karsilastirma_durumu = "hata_buyuk"
            elif oran >= 0.75:
                karsilastirma_durumu = "tebrikler"
            elif oran <= 0.15:
                karsilastirma_durumu = "hata_kucuk"

            if yuzde == 0:
                guven_araligi_metni = f"{tam_blok}.00 Blok"
                uretim_metni = f"**{tam_blok} tam blok** risksiz üretilir. Takım tam **{tam_blok}. bloğun bitiminde** tolerans sınırını aşacaktır."
            else:
                guven_araligi_metni = f"{exact_cross:.2f} Blok"
                uretim_metni = f"**{tam_blok} tam blok** risksiz üretilir. Yeni bloğa geçildiğinde kesimin **%{yuzde}'sinde** (Eksen Değeri: **{exact_cross:.2f}**) tolerans aşılır."
                
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
            'karsilastirma_durumu': karsilastirma_durumu
        }

    def plot_dashboard(self):
        if not self.scenarios:
            st.warning("Gösterilecek veri yok.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        colors = ['#E31837', '#004B87', '#d62728', '#1f77b4', '#ff7f0e']
        
        genel_max_blok = max([data['b_fut'][-1] for data in self.scenarios.values()])
        genel_max_time = max([data['b_fut'][-1] * data['cam_cycle_time'] for data in self.scenarios.values()])

        for i, (name, data) in enumerate(self.scenarios.items()):
            col = colors[i % len(colors)]
            etiket = f"{name} ({data['mat_name']})"

            st.markdown(f"""
                <div style='background-color:{col}; color:white; padding:5px 15px; border-radius:5px; display:inline-block; margin-top:15px; font-weight:bold; box-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                    📌 {name.upper()} | Alaşım: {data['mat_name']}
                </div>
            """, unsafe_allow_html=True)
            
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
            col2.metric("Tam Kırılma Noktası (Blok)", data['guven_araligi_metni'])
            col3.metric("Tam Kırılma Noktası (Zaman)", data['sure_araligi_metni'])
            
            st.info(f"🟢 **Operasyon Önerisi:** {data['uretim_metni']}")
            
            if data['veri_sayisi'] < 3:
                st.error("⚠️ **Düşük Veri Yoğunluğu:** Modele 3'ten az ölçüm girilmiştir.")
            
            if data['uzak_tahmin_uyarisi']:
                st.warning("🔭 **Aşırı Uzak Tahmin:** Uzun vadeli tahminler yanıltıcı olabilir.")

            if data['karsilastirma_durumu'] == "hata_buyuk":
                st.error(f"🛑 **Fiziksel Tutarsızlık İhtimali:** Hesaplanan süre Teorik Takım Ömrünü ({data['t_theo']:.1f} Dk) aşıyor.")
            elif data['karsilastirma_durumu'] == "tebrikler":
                st.success(f"🏆 **Mükemmel Optimizasyon:** Takım ömrünüz teorik fiziksel sınırlara çok yakın!")
            elif data['karsilastirma_durumu'] == "hata_kucuk":
                st.error(f"⚠️ **Aşırı Erken Aşınma:** Takım ömrü teorik değerin %15'inden bile daha az!")

            rmse_sinir = 10.0 if self.birim_ad == "Mikron" else 0.01
            if data['rmse_val'] > rmse_sinir:
                st.warning(f"⚠️ **Veri Anomalyası:** Sapma: {data['rmse_val']:.4f} {self.birim_ad}.")
            st.divider()

        y_limit = self.tolerance * 1.5
        for ax, title, xlabel in zip([ax1, ax2], 
                                     ["Blok Sayısına Göre Takım Aşınması", "CAM Süresine Göre Takım Aşınması"], 
                                     ["İşlenen Sütun / Blok Sayısı", "Aktif CAM İşleme Süresi (Dakika)"]):
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

MALZEMELER = {
    "Alüminyum 6061-T6": {"kc": 800, "c_taylor": 4.5e10},
    "Alüminyum 7075-T6": {"kc": 975, "c_taylor": 3.5e10},
    "Alüminyum 2024-T3": {"kc": 900, "c_taylor": 4.0e10},
    "Alüminyum 5052-H32": {"kc": 700, "c_taylor": 5.0e10},
    "Titanyum Ti-6Al-4V": {"kc": 2100, "c_taylor": 1.2e10},
    "Titanyum Ti-5Al-2.5Sn": {"kc": 2050, "c_taylor": 1.3e10},
    "Paslanmaz Çelik 304": {"kc": 2100, "c_taylor": 2.8e10},
    "17-4 PH Paslanmaz": {"kc": 2600, "c_taylor": 2.0e10},
    "AISI 4340 Alaşımlı Çelik": {"kc": 2700, "c_taylor": 1.8e10}
}

eksik_alanlar = []

with st.sidebar:
    st.header("⚙️ Genel Ayarlar")
    
    birim_secimi = st.radio("📏 Ölçüm Birimi Sistemi", ["Mikron (µm)", "Milimetre (mm)"], horizontal=True)
    is_mikron = "Mikron" in birim_secimi
    birim_ad = "Mikron" if is_mikron else "mm"
    tol_ornek = "Örn: 5" if is_mikron else "Örn: 0.005"
    cmm_ornek = "Örn: 0.2 0.5 0.8 1.2 1.6 2" if is_mikron else "Örn: 0.0002 0.0005 0.0008 0.0012 0.0016 0.0020"

    tol_siniri = st.number_input(f"Maksimum Tolerans ({birim_ad})", value=None, format="%g", placeholder=tol_ornek)
    if tol_siniri is None:
        eksik_alanlar.append("Genel Ayarlar: Maksimum Tolerans")

    senaryo_sayisi = st.number_input("Karşılaştırılacak Senaryo Sayısı", min_value=1, max_value=5, value=1, step=1)
    
    st.markdown("---")
    st.subheader("🔗 Ortak Parametre Seçimleri")
    ortak_malzeme = st.checkbox("Tüm senaryolarda ortak MALZEME kullan", value=True)
    ortak_takim = st.checkbox("Tüm senaryolarda ortak TAKIM kullan", value=True)

    genel_malzeme = None
    if ortak_malzeme:
        genel_malzeme_secimi = st.selectbox("Ortak Hammadde Seçimi", list(MALZEMELER.keys()), index=None, placeholder="Malzeme Seçiniz...")
        if genel_malzeme_secimi: genel_malzeme = MALZEMELER[genel_malzeme_secimi]

    genel_t_cap, genel_t_dis, genel_t_boy = None, None, None
    if ortak_takim:
        genel_t_cap = st.number_input("Ortak Takım Çapı (D) [mm]", value=None, min_value=1, step=1, placeholder="Örn: 6")
        genel_t_dis = st.number_input("Ortak Takım Diş Sayısı (z)", value=None, min_value=1, step=1, placeholder="Örn: 4")
        genel_t_boy = st.number_input("Ortak Takım Kesme Boyu (Lc) [mm]", value=None, min_value=1, step=1, placeholder="Örn: 24")

st.markdown(f"### 📋 Test Verisi Girişi ({senaryo_sayisi} Senaryo)")
sekmeler = st.tabs([f"{i+1}. Senaryo" for i in range(senaryo_sayisi)])
senaryo_verileri = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim = st.text_input(f"Senaryo Adı", value=f"Senaryo {i+1}", key=f"isim_{i}")
        
        # --- SAĞ BOŞLUĞU DOLDURAN 3 SÜTUNLU YAPI ---
        colA, colB, colC = st.columns([1.3, 1.3, 1])
        
        with colA:
            st.markdown("**Malzeme ve Takım Ayarları**")
            if not ortak_malzeme:
                m_secim = st.selectbox("Hammadde Seçimi", list(MALZEMELER.keys()), index=None, placeholder="Malzeme Seçiniz...", key=f"mat_{i}")
                s_malzeme = MALZEMELER[m_secim] if m_secim else None
                s_malzeme_isim = m_secim
                if s_malzeme is None: eksik_alanlar.append(f"{isim}: Hammadde Seçimi")
            else:
                s_malzeme = genel_malzeme
                s_malzeme_isim = genel_malzeme_secimi
                if s_malzeme is None: eksik_alanlar.append(f"{isim}: Ortak Hammadde Seçimi")

            if not ortak_takim:
                t_cap = st.number_input("Takım Çapı (D) [mm]", value=None, min_value=1, step=1, placeholder="Örn: 6", key=f"tcap_{i}")
                t_dis = st.number_input("Takım Diş Sayısı (z)", value=None, min_value=1, step=1, placeholder="Örn: 4", key=f"tdis_{i}")
                t_boy = st.number_input("Takım Kesme Boyu (Lc) [mm]", value=None, min_value=1, step=1, placeholder="Örn: 24", key=f"tboy_{i}")
                if t_cap is None: eksik_alanlar.append(f"{isim}: Takım Çapı")
                if t_dis is None: eksik_alanlar.append(f"{isim}: Takım Diş Sayısı")
                if t_boy is None: eksik_alanlar.append(f"{isim}: Takım Kesme Boyu")
            else:
                t_cap, t_dis, t_boy = genel_t_cap, genel_t_dis, genel_t_boy
                if t_cap is None: eksik_alanlar.append(f"{isim}: Ortak Takım Çapı")
                if t_dis is None: eksik_alanlar.append(f"{isim}: Ortak Takım Diş Sayısı")
                if t_boy is None: eksik_alanlar.append(f"{isim}: Ortak Takım Kesme Boyu")

        with colB:
            st.markdown("**Kesme ve Ölçüm Parametreleri**")
            vc = st.number_input("Kesme Hızı (Vc) [m/min]", value=None, min_value=1, step=1, placeholder="Örn: 400", key=f"vc_{i}")
            if vc is None: eksik_alanlar.append(f"{isim}: Kesme Hızı (Vc)")
            fz = st.number_input("İlerleme (fz) [mm/diş]", value=None, format="%g", placeholder="Örn: 0.08", key=f"fz_{i}")
            if fz is None: eksik_alanlar.append(f"{isim}: İlerleme (fz)")
            ap = st.number_input("Eksenel Derinlik (ap) [mm]", value=None, format="%g", placeholder="Örn: 5.0", key=f"ap_{i}")
            if ap is None: eksik_alanlar.append(f"{isim}: Eksenel Derinlik (ap)")
            ae = st.number_input("Radyal Derinlik (ae) [mm]", value=None, format="%g", placeholder="Örn: 5.0", key=f"ae_{i}")
            if ae is None: eksik_alanlar.append(f"{isim}: Radyal Derinlik (ae)")
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                cam_dk = st.number_input("Dakika", value=None, placeholder="Örn: 2", min_value=0, step=1, key=f"cam_dk_{i}")
                if cam_dk is None: eksik_alanlar.append(f"{isim}: İşleme Süresi (Dakika)")
            with t_col2:
                cam_sn = st.number_input("Saniye (Opsiyonel)", value=None, placeholder="Örn: 15", min_value=0, max_value=59, step=1, key=f"cam_sn_{i}")
            
            cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None
            cmm_str = st.text_input(f"CMM Verileri ({birim_ad}, Boşluklu)", value="", placeholder=cmm_ornek, key=f"cmm_{i}")
            if not cmm_str: eksik_alanlar.append(f"{isim}: CMM Verileri")

        # --- YENİ SAĞ PANEL (HTML formatlı, hatası giderilmiş) ---
        with colC:
            tol_text = f"{tol_siniri} {birim_ad}" if tol_siniri else "Belirlenmedi"
            mat_text = f"<b>Alaşım:</b> {s_malzeme_isim}<br><b>Kc:</b> {s_malzeme['kc']} MPa<br><b>Taylor Sabiti:</b> {s_malzeme['c_taylor']:.1e}" if s_malzeme else "Malzeme Bekleniyor..."
            
            st.markdown(f"""
            <div class='right-panel-card'>
                <h4 style="margin-top: 0; color: #004B87; border-bottom: 2px solid rgba(136,136,136,0.2); padding-bottom: 10px;">🧠 Akıllı Fizik Motoru</h4>
                <p style="font-size: 14px; margin-bottom: 5px;"><b>Aktif Tolerans:</b></p>
                <div style="color: #E31837; font-size: 18px; font-weight: bold; margin-bottom: 15px;">{tol_text}</div>
                
                <p style="font-size: 14px; margin-bottom: 5px;"><b>Fiziksel Parametreler:</b></p>
                <div style="background-color: rgba(0, 75, 135, 0.08); padding: 10px; border-radius: 5px; font-size: 13px; margin-bottom: 15px;">
                    {mat_text}
                </div>
                
                <div style="font-size: 12px; color: gray; border-left: 2px solid #E31837; padding-left: 10px; font-style: italic;">
                    Sistem, anlık talaş kalınlığını hesaplar ve efektif kesme kuvvetini Taylor denklemi ile harmanlayarak Kırılma Ufkunu belirler.
                </div>
            </div>
            """, unsafe_allow_html=True)

        senaryo_verileri.append({
            "isim": isim, "mat_isim": s_malzeme_isim, "mat_data": s_malzeme,
            "t_cap": t_cap, "t_dis": t_dis, "t_boy": t_boy,
            "vc": vc, "fz": fz, "ap": ap, "ae": ae, "cam_sure": cam_sure, "cmm_str": cmm_str
        })

st.markdown("---")

if st.button("🚀 Takım Ömrü Tahminini Başlat", use_container_width=True, type="primary"):
    if len(eksik_alanlar) > 0:
        hata_metni = "\n".join([f"- {alan}" for alan in list(set(eksik_alanlar))])
        st.error(f"⚠️ Lütfen analizi başlatmadan önce aşağıdaki eksik bilgileri doldurunuz:\n\n{hata_metni}")
    else:
        try:
            system = AI_ToolLife(tolerance=tol_siniri, birim_ad=birim_ad)
            for d in senaryo_verileri:
                cmm_vals = [float(x) for x in d["cmm_str"].replace(',', ' ').split()]
                system.add_scenario(d["isim"], d["mat_isim"], d["mat_data"]['kc'], d["mat_data"]['c_taylor'], d["t_cap"], d["t_dis"], d["t_boy"], d["vc"], d["fz"], d["ap"], d["ae"], list(range(1, len(cmm_vals) + 1)), cmm_vals, d["cam_sure"])
            system.plot_dashboard()
        except Exception as e:
            st.error(f"CMM Verisi veya sayısal format hatası: {e}. Lütfen sadece sayıları ve boşlukları kullandığınızdan emin olun.")
