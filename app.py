import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
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
    **Bu sistem, İstatistiksel Analizler kullanarak parça üzerindeki kritik bölgelerin aşınma ufuklarını tahmin eder.**

    ###  🛠️  Nasıl Kullanılır?
    1. **Veri Giriş:** Sol menüden CMM dosya yükleme (PDF/Otonom) yöntemini seçin ve dosyaları yükleyin.
    2. **Çoklu Eşleştirme:** Operatör isim değiştirmiş olabileceği için, izlenecek bölgeye ait **tüm isim varyasyonlarını** açılır listeden (multiselect) seçin. (Örn: `1_PROFILE` ve `1_0.1 BILGI...`).
    3. **Analiz:** 'Kestirim Analizini Başlat' butonuna basın ve 3-fazlı (S-Curve) grafiklerini inceleyin.
    """)

if st.session_state.ilk_giris:
    rehber_dialog()
    st.session_state.ilk_giris = False

# --- 2. CSS TEMA ---
st.markdown("""
<style>
header[data-testid="stHeader"] { background: linear-gradient(90deg, #004B87, #E31837) !important; height: 4px !important; }
h1, h2, h3, h4 { color: #004B87 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
[data-testid="metric-container"] { border: 1px solid rgba(136, 136, 136, 0.2); padding: 15px; border-radius: 8px; background-color: rgba(248, 249, 250, 0.05); box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
[data-testid="stMetricValue"] { color: #004B87 !important; font-weight: 800; }
div.stButton > button:first-child { background: linear-gradient(90deg, #004B87, #0066cc); color: #FFFFFF; font-weight: bold; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
div.stButton > button:first-child:hover { background: linear-gradient(90deg, #E31837, #ff3333); transform: scale(1.02); }
[data-testid="stSidebar"] { border-right: 3px solid #E31837; }
[data-testid="stSidebar"]::before { content: "REMOVE BEFORE FLIGHT"; display: block; background-color: #E31837; color: white; font-family: monospace; font-weight: bold; text-align: center; padding: 6px; letter-spacing: 1.5px; margin-bottom: 20px; border-radius: 0 0 5px 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: left; background-color: #E31837; color: white; display: inline-block; padding: 3px 12px; font-family: monospace; font-weight: bold; border-radius: 4px; font-size: 13px; box-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>by Fuat Arıkan</div>", unsafe_allow_html=True)

col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'> 🛠️  LIFT-UP: Kestirimci Bakım Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='height: 3px; background: linear-gradient(90deg, transparent, #004B87 30%, #E31837 70%, transparent); border: none; margin-top: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)

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

# --- 4. GELİŞMİŞ PDF MOTORU (Tolerans Miras Alma Özelliği Aktif) ---
def is_float_loose(s):
    s_temiz = s.replace('°', '').strip()
    return bool(re.match(r'^[-+]?[0-9]*[.,]?[0-9]+$', s_temiz))

def extract_pdf_data_advanced(file):
    rows_list = []
    # Varsayılan başlangıç toleransları
    son_ust_tol = 0.100
    son_alt_tol = -0.100
    
    try:
        with pdfplumber.open(file) as pdf:
            in_table = False
            for sayfa in pdf.pages:
                metin = sayfa.extract_text(layout=True)
                if not metin: continue
                
                for satir in metin.split('\n'):
                    satir = satir.strip()
                    if not satir: continue
                        
                    if "Measured" in satir and "Nominal" in satir and "Name" in satir:
                        in_table = True
                        continue
                    if not in_table: continue
                    if "Page " in satir and " of " in satir:
                        in_table = False
                        continue
                    if "--- PAGE" in satir or "ZEISS" in satir or "TOMTAS" in satir or "HAVACILIK" in satir:
                        continue
                        
                    parts = satir.split()
                    first_num_idx = -1
                    
                    for i in range(1, len(parts)):
                        if is_float_loose(parts[i]):
                            first_num_idx = i
                            break
                            
                    if first_num_idx > 0:
                        name = " ".join(parts[:first_num_idx]).strip()
                        # Metadata ve sayfa altı bilgilerini engelle
                        if name.startswith(("Program", "Revision", "Order", "Lot", "Text", "Kalibrasyon", "CMM", "Operator", "Date", ")", "INFORMATION", "Part")):
                            continue
                            
                        remaining = parts[first_num_idx:]
                        numbers = []
                        
                        for r in remaining:
                            if is_float_loose(r):
                                numbers.append(r.replace('°', '').replace(',', '.'))
                                
                        if len(numbers) >= 5:
                            meas = float(numbers[0])
                            nom = float(numbers[1])
                            son_ust_tol = float(numbers[2]) # TOLERANS HAFIZAYA ALINIYOR (Miras Alma)
                            son_alt_tol = float(numbers[3])
                            dev = float(numbers[4])
                        elif len(numbers) == 3:
                            meas = float(numbers[0])
                            nom = float(numbers[1])
                            dev = float(numbers[2])
                            # Tolerans boş! Bir üstteki tolerans miras alınıyor (son_ust_tol korunuyor)
                        elif len(numbers) == 1:
                            meas = float(numbers[0])
                            nom = 0.0
                            dev = meas
                        else:
                            if numbers:
                                meas = float(numbers[0])
                                nom = float(numbers[1]) if len(numbers) > 1 else 0.0
                                dev = float(numbers[-1])
                            else: continue
                                
                        rows_list.append({
                            "Olcum_Adi": name,
                            "Olculen_Deger": meas,
                            "Nominal": nom,
                            "Sapma": abs(dev),
                            "Ust_Tolerans": son_ust_tol,
                            "Alt_Tolerans": son_alt_tol
                        })
    except Exception as e:
        st.error(f"PDF Okuma Hatası: {e}")
    return pd.DataFrame(rows_list)

def clean_cmm_data(df):
    sutun_haritasi = {
        'Name': 'Olcum_Adi', 'Dimension': 'Olcum_Adi', 'Measured value': 'Olculen_Deger',
        'Nominal value': 'Nominal', '+Tol': 'Ust_Tolerans', '-Tol': 'Alt_Tolerans', 'Deviation': 'Sapma'
    }
    df.rename(columns=sutun_haritasi, inplace=True)
    return df

# --- 5. 3-FAZLI PÜRÜZSÜZ S-EĞRİSİ MOTORU (Noktaları Yakalama) ---
class AI_ToolLife:
    def __init__(self, birim_ad):
        self.birim_ad = birim_ad
        self.scenarios = {}

    def uclu_faz_modeli(self, blocks, wear_data, future_blocks, tolerance):
        x = np.array(blocks, dtype=float)
        y = np.array(wear_data, dtype=float)
        y_fut = np.zeros(len(future_blocks))
        
        if len(x) < 2: return np.full(len(future_blocks), y[0] if len(y)>0 else 0)

        # Benzersiz X değerlerini al (Interpolation hatalarını önlemek için)
        _, u_idx = np.unique(x, return_index=True)
        x_u = x[u_idx]
        y_u = y[u_idx]

        # Noktaların İÇİNDEN GEÇEN akıllı eğri (Pchip Interpolator)
        try:
            if len(x_u) >= 3:
                interp_func = PchipInterpolator(x_u, y_u)
            else:
                interp_func = interp1d(x_u, y_u, kind='linear', fill_value='extrapolate')
        except:
            interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')

        # Kararlı Fazın (Faz 2) İlerleme İvmesi
        try:
            egim, _ = np.polyfit(x[-3:] if len(x) >= 3 else x, y[-3:] if len(x) >= 3 else y, 1)
        except:
            egim = 0.001 
        if egim <= 0: egim = 0.001 

        uyari_siniri = tolerance * 0.70

        for i, bx in enumerate(future_blocks):
            if bx <= x[-1]:
                # FAZ 1 & 2: Gerçek CMM noktalarının rotasını kopyala
                val = interp_func(bx) if bx >= x[0] else y[0]
            else:
                # FAZ 2 (Gelecek): Lineer Eğim
                val = y[-1] + egim * (bx - x[-1])
                
                # FAZ 3 (Kırılma): Toleransın %70'inden sonra Parabolik Şahlanma (S-Curve)
                if val > uyari_siniri:
                    carpan = (val - uyari_siniri) / (tolerance - uyari_siniri)
                    faz3_siddeti = 0.5 * (carpan ** 3) * tolerance 
                    val += faz3_siddeti
            y_fut[i] = val
            
        # Fizik kuralları: Aşınma eksiye düşemez ve kümülatiftir
        y_fut = np.maximum(y_fut, 0)
        y_fut = np.maximum.accumulate(y_fut)
        
        return y_fut

    def add_scenario(self, name, blocks, wear_data, tolerance, cam_cycle_time=None):
        if len(blocks) < 2:
            raise ValueError(f"{name} analizi için en az 2 ardışık parça ölçümü gerekiyor.")
            
        max_blok = max(blocks)
        veri_sayisi = len(blocks)
        
        # Grafiğin daha "Kıvrımlı/Pürüzsüz" görünmesi için noktaları yoğunlaştır
        future_blocks = np.linspace(min(blocks), max_blok * 10 + 50, 400)
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
            
        limit_idx = np.where(future_blocks >= grafik_son_blok)[0]
        kesme_noktasi = limit_idx[0] if len(limit_idx) > 0 else len(future_blocks)
        
        self.scenarios[name] = {
            'b_raw': blocks, 'y_raw': wear_data, 
            'b_fut': future_blocks[:kesme_noktasi], 'y_fut': y_fut_array[:kesme_noktasi],
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
            # Arka Plan Şeritleri
            ax.axhspan(0, uyari_siniri, facecolor='#d4edda', alpha=0.4)
            ax.axhspan(uyari_siniri, tol, facecolor='#fff3cd', alpha=0.5)
            ax.axhspan(tol, tol * 1.5, facecolor='#f8d7da', alpha=0.4)

            ax.axhline(tol, color='#dc3545', linewidth=3, linestyle='--', label=f"Maksimum Tolerans ({tol} {self.birim_ad})")
            ax.axhline(uyari_siniri, color='#ffc107', linewidth=2.5, linestyle='--', label=f"Erken Uyarı Eşiği ({uyari_siniri:.3f} {self.birim_ad})")

            ax.plot(x_fut, data['y_fut'], color='#004B87', linestyle='-', linewidth=4, zorder=4, label="3-Fazlı Aşınma Eğrisi")
            ax.scatter(x_raw, data['y_raw'], color='#E31837', s=160, zorder=5, edgecolor='white', linewidth=2, label='CMM Ölçümleri')
            
            ax.set_title(f"Kritik Bölge: {name.upper()}", fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Boyutsal Sapma [{self.birim_ad}]", fontsize=11, fontweight='bold')
            ax.set_ylim(0.0, tol * 1.3)
            if len(x_fut) > 0: ax.set_xlim(0, x_fut[-1] * 1.05)
            
            # Göstergeyi (Legend) alt kısma, grafiğin dışına taşıdık ki çizgileri kapatmasın
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10, frameon=True, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

        fig.tight_layout(pad=4.0)
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
    st.info("💡 **Otonom Mod:** PDF raporlarını sırayla sürükleyiniz. Tolerans değerleri dosyadan otomatik çekilir.")
    yuklenen_dosyalar = st.file_uploader("CMM Raporlarını Klasör Halinde Sürükleyin", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if yuklenen_dosyalar:
        veri_listesi = []
        parca_sayaci = 1
        for dosya in yuklenen_dosyalar:
            try:
                if dosya.name.lower().endswith('.pdf'):
                    df_temp = extract_pdf_data_advanced(dosya)
                else:
                    df_temp = clean_cmm_data(pd.read_excel(dosya))
                
                if not df_temp.empty:
                    df_temp['Parca_Sira'] = parca_sayaci
                    veri_listesi.append(df_temp)
                    parca_sayaci += 1
            except Exception as e:
                st.error(f"Dosya okuma hatası ({dosya.name}): {e}")
                
        if veri_listesi:
            df_ana = pd.concat(veri_listesi, ignore_index=True)

# --- 8. PARAMETRE GİRİŞİ VE ÇOKLU EŞLEŞTİRME ---
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
                    
                    # ÇOKLU SEÇİM KUTUSU (MULTİSELECT): İsim değişimlerine karşı koruma
                    secilen_olcumler = st.multiselect("📏 Dosyadaki Karşılığı (Birden fazla seçilebilir)", olcumler, key=f"geo_{i}", placeholder="Bölge Seçin...")
                    
                    if not secilen_olcumler: eksik_alanlar.append(f"{isim}: Bölge Seçimi")
                    aykiri_filtre = st.checkbox("Outlier Sapmaları Temizle", value=True, key=f"outlier_{i}")
                    cmm_str, bolge_toleransi = "", None
                    
                    if secilen_olcumler:
                        df_sub = df_ana[df_ana['Olcum_Adi'].isin(secilen_olcumler)]
                        if 'Ust_Tolerans' in df_sub.columns and pd.notna(df_sub['Ust_Tolerans'].iloc[0]):
                            bolge_toleransi = df_sub['Ust_Tolerans'].iloc[0]
                else:
                    st.info("Eşleştirme listesi için geçerli PDF bekleniyor.")
                    eksik_alanlar.append(f"{isim}: CMM Dosyası")
                    secilen_olcumler, aykiri_filtre, cmm_str, bolge_toleransi = [], False, "", None
            else:
                st.markdown("**Manuel Ölçüm Girişi**")
                cmm_str = st.text_input(f"Aşınma Değerleri ({birim_ad})", key=f"cmm_{i}")
                if not cmm_str: eksik_alanlar.append(f"{isim}: Manuel CMM Serisi")
                secilen_olcumler, aykiri_filtre, bolge_toleransi = [], False, tol_siniri

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
            if veri_giris_modu == "CMM Dosyası Yükle (PDF/Otonom)" and secilen_olcumler:
                st.success(f"✅ Seçilen bölge(ler) için {len(df_sub)} adet ardışık veri çekildi.")
                if bolge_toleransi:
                    st.info(f"📏 Bu bölgenin spesifik toleransı dosyadan **{bolge_toleransi} {birim_ad}** olarak ayarlandı.")
                else:
                    st.warning("⚠️ Dosyada Tolerans bulunamadı, manuel giriniz.")
                    bolge_toleransi = st.number_input(f"Manuel Tolerans ({birim_ad})", min_value=0.001, value=0.100, key=f"man_tol_{i}")

        senaryo_verileri.append({
            "isim": isim, "cmm_str": cmm_str, "secilen_olcumler": secilen_olcumler, 
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
                    df_sub = df_ana[df_ana['Olcum_Adi'].isin(d["secilen_olcumler"])].copy()
                    
                    if d["aykiri_filtre"]:
                        mean_val = df_sub['Sapma'].mean()
                        std_val = df_sub['Sapma'].std()
                        if std_val > 0:
                            df_sub = df_sub[np.abs((df_sub['Sapma'] - mean_val) / std_val) < 3]
                            
                    raw_vals = df_sub['Sapma'].tolist()
                    raw_blocks = df_sub['Parca_Sira'].tolist() if 'Parca_Sira' in df_sub.columns else list(range(1, len(raw_vals) + 1))
                    
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
