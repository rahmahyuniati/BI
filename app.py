import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from base64 import b64encode
from itertools import combinations
from collections import Counter
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Beryl Coffee Dashboard", layout="wide")

# 2. FUNGSI LOGO & DATA
def get_base64(bin_file):
    try:
        with open(bin_file, "rb") as f:
            return b64encode(f.read()).decode()
    except: return ""

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("DatasetBerylCoffe_Chill.csv") 
    df['Total Penjualan (Rp)'] = pd.to_numeric(df['Total Penjualan (Rp)'].astype(str).str.replace(r'[^\d]', '', regex=True))
    df['Waktu Order'] = pd.to_datetime(df['Waktu Order'])
    df['Tanggal Order'] = pd.to_datetime(df['Waktu Order'].dt.date)
    # Pastikan Jam Order tersedia untuk analisis pola waktu
    if 'Jam Order' not in df.columns:
        df['Jam Order'] = df['Waktu Order'].dt.time
    return df

# Inisialisasi
df = load_and_clean_data()
logo_base64 = get_base64("WhatsApp Image 2025-12-28 at 19.06.13.jpeg")

# HARI
df['Hari'] = df['Tanggal Order'].dt.day_name()
hari_transaksi = df['Hari'].value_counts()

#  Menyiapkan data transaksi (pecah string produk menjadi list)
transaksi_items = df['Produk'].apply(lambda x: [item.strip() for item in str(x).split(',') if item.strip()])

#  Mencari kombinasi 2 menu
kombinasi_menu = []
for item_list in transaksi_items:
    if len(item_list) >= 2:
        kombinasi_menu.extend(combinations(sorted(set(item_list)), 2))

#  Menghitung frekuensi dan membuat DataFrame 'top_kombinasi'
frekuensi_kombinasi = Counter(kombinasi_menu)
top_kombinasi = pd.DataFrame(
    frekuensi_kombinasi.most_common(10),
    columns=['Kombinasi Menu', 'Frekuensi']
)

#  Membuat kolom string untuk keperluan visualisasi (agar tidak muncul sebagai tuple)
top_kombinasi['Kombinasi Menu String'] = top_kombinasi['Kombinasi Menu'].apply(lambda x: ', '.join(x))

# --- Penyiapan Variabel EDA 5 ---
# Expand produk untuk menghitung kontribusi per item
df_expanded = df.assign(Produk=df['Produk'].str.split(',')).explode('Produk')
df_expanded['Produk'] = df_expanded['Produk'].str.strip()
menu_sales = df_expanded.groupby('Produk')['Total Penjualan (Rp)'].sum().sort_values(ascending=False)
menu_terlaris = menu_sales.index[0]

# --- Penyiapan Data Tab 5 ---
# 1. Hitung kontribusi per menu
df_expanded = df.assign(Produk=df['Produk'].str.split(',')).explode('Produk')
df_expanded['Produk'] = df_expanded['Produk'].str.strip()
menu_sales = df_expanded.groupby('Produk')['Total Penjualan (Rp)'].sum().sort_values(ascending=False)
menu_terlaris = menu_sales.index[0]

# 2. Hitung tren untuk menu terlaris (Untuk grafik kedua)
tren_menu_top = (
    df_expanded[df_expanded['Produk'] == menu_terlaris]
    .groupby('Tanggal Order')['Total Penjualan (Rp)']
    .sum()
)

# --- Penyiapan Variabel EDA 6 ---
# Ekstrak bulan dan urutkan
df['bulan'] = df['Tanggal Order'].dt.month_name()
list_bulan = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
penjualan_bulanan = df.groupby('bulan')['Total Penjualan (Rp)'].sum().reindex(list_bulan).dropna()


# --- Penyiapan Variabel EDA 7 ---
# Group by Jenis Order
perilaku_order_jenis = df.groupby('Jenis Order').agg({
    'Total Penjualan (Rp)':'sum'
}).rename(columns={'Total Penjualan (Rp)':'Total Belanja'}).sort_values(by='Total Belanja', ascending=False)

# Hitung jumlah item per transaksi
df['Jumlah_Item_Transaksi'] = df['Produk'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

# --- Penyiapan Variabel EDA 8 ---
# Moving Average 3 Bulan
tren_penjualan = penjualan_bulanan.to_frame(name='Penjualan')
tren_penjualan['MA_3'] = tren_penjualan['Penjualan'].rolling(window=3).mean()



# --- 3. HITUNG METRIK KPI ---
total_revenue = df['Total Penjualan (Rp)'].sum()
total_trans = len(df)
avg_trans = total_revenue / total_trans if total_trans > 0 else 0
product_counts = df["Produk"].str.split(",", expand=True).stack().value_counts()
top_prod = product_counts.index[0] if not product_counts.empty else "N/A"

# Perhitungan Prime Time otomatis untuk KPI
try:
    df['Jam_Int'] = pd.to_datetime(df['Jam Order'], format='%H:%M:%S').dt.hour
except:
    df['Jam_Int'] = df['Jam Order'].apply(lambda x: x.hour if hasattr(x, 'hour') else None)
jam_transaksi_counts = df['Jam_Int'].value_counts().sort_index()
peak_hour_val = jam_transaksi_counts.idxmax()
peak_hour = f"{peak_hour_val}:00 WIB"

# --- 4. TAMPILAN HEADER & TOP 5 MENU (MODIFIED) ---

# Mengambil Top 5 Menu dari data
top_5_products = product_counts.head(5)
top_5_names = top_5_products.index.tolist()
top_5_values = top_5_products.values.tolist()

st.markdown(f"""
<style>
    .header-card {{ background: linear-gradient(135deg, #C28A34, #A67329); padding: 40px 50px; border-radius: 25px; margin-bottom: 30px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); }}
    .header-top-row {{ display: flex; align-items: center; gap: 25px; margin-bottom: 10px; }}
    .logo-img {{ width: 80px; height: 80px; object-fit: cover; border-radius: 50%; border: 3px solid white; }}
    .header-title {{ font-size: 42px; font-weight: 800; color: white !important; margin: 0; letter-spacing: -1px; }}
    .header-subtitle {{ color: #FDFCF0; font-size: 18px; font-weight: 400; opacity: 0.95; margin-bottom: 20px; }}
    .badge {{ background-color: rgba(255,255,255,0.2); color: white; padding: 6px 16px; border-radius: 50px; font-size: 13px; font-weight: 600; margin-right: 8px; border: 1px solid rgba(255,255,255,0.3); }}
    
    /* Judul Top 5 Menu */
    .section-title {{ font-size: 24px; font-weight: 800; color: #2D2417; margin-bottom: 15px; margin-top: 10px; padding-left: 5px; border-left: 5px solid #C28A34; }}

    /* Wrapper Kotak */
    .kpi-wrapper {{ display: flex; gap: 15px; margin-bottom: 40px; }}
    
    /* Kotak dibuat lebih panjang ke bawah (min-height ditambahkan) */
    .beryl-card-long {{ 
        flex: 1; 
        background: white; 
        border-radius: 20px; 
        padding: 30px 20px; 
        min-height: 200px; 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
        align-items: center; 
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); 
        border-bottom: 8px solid #C28A34;
        transition: transform 0.3s;
    }}
    .beryl-card-long:hover {{ transform: scale(1.02); }}
    
    .rank-label {{ font-size: 12px; font-weight: 700; color: #C28A34; text-transform: uppercase; margin-bottom: 10px; }}
    .menu-name {{ font-size: 20px; font-weight: 850; color: #222; line-height: 1.2; margin-bottom: 10px; }}
    .menu-total {{ font-size: 14px; color: #666; font-weight: 600; }}
</style>

<div class="header-card">
    <div class="header-top-row">
        <img src="data:image/jpeg;base64,{logo_base64}" class="logo-img">
        <h1 class="header-title"> Beryl Coffee & Chill</h1>
    </div>
    <div class="header-subtitle">Analisis Pola Pembelian & Prediksi Tren Penjualan</div>
    <div><span class="badge">EDA</span><span class="badge">FP-Growth</span><span class="badge"> DATA-DRIVEN DECISION </span></div>
</div>

<div class="section-title">🏆 Top 5 Menu Terlaris</div>

<div class="kpi-wrapper">
    <div class="beryl-card-long">
        <div class="rank-label">Rank 1</div>
        <div class="menu-name">{top_5_names[0]}</div>
        <div class="menu-total">{top_5_values[0]} Transaksi</div>
    </div>
    <div class="beryl-card-long">
        <div class="rank-label">Rank 2</div>
        <div class="menu-name">{top_5_names[1]}</div>
        <div class="menu-total">{top_5_values[1]} Transaksi</div>
    </div>
    <div class="beryl-card-long">
        <div class="rank-label">Rank 3</div>
        <div class="menu-name">{top_5_names[2]}</div>
        <div class="menu-total">{top_5_values[2]} Transaksi</div>
    </div>
    <div class="beryl-card-long">
        <div class="rank-label">Rank 4</div>
        <div class="menu-name">{top_5_names[3]}</div>
        <div class="menu-total">{top_5_values[3]} Transaksi</div>
    </div>
    <div class="beryl-card-long">
        <div class="rank-label">Rank 5</div>
        <div class="menu-name">{top_5_names[4]}</div>
        <div class="menu-total">{top_5_values[4]} Transaksi</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Kode ini harus ada sebelum baris 210
from itertools import combinations
from collections import Counter

transaksi_items = df['Produk'].apply(lambda x: [item.strip() for item in str(x).split(',') if item.strip()])
kombinasi_menu = []
for item_list in transaksi_items:
    if len(item_list) >= 2:
        kombinasi_menu.extend(combinations(sorted(set(item_list)), 2))

frekuensi_kombinasi = Counter(kombinasi_menu)
top_kombinasi = pd.DataFrame(frekuensi_kombinasi.most_common(10), columns=['Kombinasi Menu', 'Frekuensi'])
top_kombinasi['Kombinasi Menu String'] = top_kombinasi['Kombinasi Menu'].apply(lambda x: ', '.join(x))


# =====================================================
# ANALISIS DATA EKSPLORATIF (EDA)
# =====================================================
st.header("🎯 Analisis Data Eksploratif (EDA)")

# --- TAB 1 & 2 ---
st.subheader("Analisis Produk & Tren Penjualan")
tab1, tab2 = st.tabs(["📋 Pola Pembelian", "📈 Tren Penjualan Harian"])

with tab1:
    st.markdown("##### 1. Bagaimana pola pembelian produk berdasarkan transaksi pelanggan?")
    col1, _ = st.columns([0.5, 0.5])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(3.5, 2))
        product_counts.head(10).plot(kind="bar", color='#C28A34', ax=ax1)
        ax1.tick_params(axis='both', labelsize=5)
        ax1.set_ylabel("Jumlah Pembelian", fontsize=6)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig1)
    st.info(f"**💡 Insight:** **{top_prod}** menjadi produk dengan frekuensi pembelian tertinggi, yang mengindikasikan bahwa produk tersebut sering dibeli sebagai pelengkap bersama produk lain. Selain itu, produk makanan dan minuman utama seperti Nasi Goreng, Florana, Matcha, dan Chocolate memiliki tingkat pembelian yang relatif tinggi dan berdekatan, menunjukkan preferensi pelanggan terhadap menu inti. Sementara itu, produk seperti Vanilla, Donat, dan Kentang Goreng memiliki frekuensi pembelian yang lebih rendah, namun tetap berkontribusi dalam transaksi sebagai produk tambahan. Secara keseluruhan, pola pembelian tidak terdistribusi secara merata, di mana sebagian kecil produk memberikan kontribusi besar terhadap total transaksi pelanggan.")

with tab2:
    st.markdown("##### 2. Bagaimana tren penjualan Beryl Coffee & Chill dari waktu ke waktu?")
    
    # Memberi jarak sedikit di bawah judul
    st.write("") 

    # Menggunakan rasio [0.5, 0.5] agar grafik tetap mungil
    col2, _ = st.columns([0.5, 0.5])
    
    with col2:
        daily_sales = df.groupby("Tanggal Order")["Total Penjualan (Rp)"].sum()
        
        # Ukuran mungil (3.5 x 2) agar konsisten dengan tab lain
        fig2, ax2 = plt.subplots(figsize=(3.5, 2))
        ax2.plot(daily_sales.index, daily_sales.values, color='#A67329', marker='o', markersize=2, linewidth=0.8)
        
        # Penyesuaian teks sumbu
        ax2.tick_params(axis='both', labelsize=5)
        ax2.set_ylabel("Total Penjualan", fontsize=6)
        plt.xticks(rotation=45, ha='right', fontsize=5)
        
        # Menghapus margin berlebih
        plt.tight_layout()
        st.pyplot(fig2) 
    
    # Memberi jarak antara grafik dan kotak insight
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Insight khusus Tab 2 (Tanpa variabel produk dari Tab 1)
    st.info("""
    **💡 Insight:** Setiap titik pada grafik penjualan Beryl Coffee & Chill selama periode September 2025 hingga awal Desember 2025 merepresentasikan total penjualan yang dihasilkan pada satu tanggal tertentu, sehingga variasi nilai yang terlihat mencerminkan perubahan aktivitas transaksi harian. 

    Pada beberapa tanggal tertentu terlihat adanya lonjakan penjualan yang cukup tinggi, yang mengindikasikan hari-hari dengan permintaan pelanggan yang lebih besar. Sebaliknya, terdapat pula beberapa tanggal dengan nilai penjualan yang relatif lebih rendah. Namun demikian, sepanjang periode pengamatan dari September hingga Desember 2025, tidak terlihat adanya tren penurunan penjualan yang berlangsung secara konsisten. Secara keseluruhan, kinerja penjualan Beryl Coffee & Chill dapat dikatakan **relatif stabil dalam jangka menengah**.
    """)
    # Tambahkan satu atau beberapa <br> untuk memberi jarak ke bawah
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
        
# --- TAB 3 & 4 ---
st.subheader("Analisis Waktu & Kombinasi Menu")
tab3, tab4 = st.tabs(["🕒 Puncak Transaksi (Jam)", "🤝 Kombinasi Menu Populer"])

with tab3:
    st.markdown("##### 3. Apakah terdapat waktu tertentu dengan tingkat penjualan tertinggi?")
    
    # Membuat kolom dengan rasio 0.6 (lebar visual) dan 0.4 (ruang kosong) agar grafik tidak melebar
    col3, _ = st.columns([0.6, 0.4]) 
    
    with col3:
        df['Jam_Int'] = df['Jam Order'].apply(lambda x: x.hour if hasattr(x, 'hour') else None)
        jam_transaksi = df['Jam_Int'].value_counts().sort_index()
        
        # Perkecil figsize (misal: lebar 4, tinggi 2.5)
        fig3, ax3 = plt.subplots(figsize=(4, 2.5)) 
        ax3.plot(jam_transaksi.index, jam_transaksi.values, marker='o', color='#A67329', linewidth=1, markersize=3)
        ax3.fill_between(jam_transaksi.index, jam_transaksi.values, color='#C28A34', alpha=0.2)
        
        ax3.set_xticks(range(0, 24, 2)) # Menampilkan tiap 2 jam agar tidak sesak
        ax3.tick_params(axis='both', labelsize=6)
        ax3.set_xlabel("Jam Operasional", fontsize=7)
        ax3.set_ylabel("Jumlah Transaksi", fontsize=7)
        ax3.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig3)
        
    st.info("""**💡 Insight:** Berdasarkan grafik jumlah transaksi berdasarkan jam, terdapat waktu tertentu dengan tingkat penjualan tertinggi di Beryl Coffee & Chill. Aktivitas penjualan mulai meningkat pada pagi menjelang siang, kemudian relatif stabil pada siang hari. Puncak penjualan terjadi pada sore hingga malam hari, dengan jumlah transaksi tertinggi tercatat sekitar pukul 20.00. Setelah mencapai puncak tersebut, jumlah transaksi mulai menurun pada jam-jam berikutnya. Pola ini menunjukkan bahwa periode malam hari merupakan waktu paling ramai, sehingga menjadi jam strategis bagi operasional dan strategi penjualan Beryl Coffee & Chill.""")

with tab4:
    st.markdown("##### 4. Kombinasi menu apa yang paling sering dibeli pelanggan dalam satu transaksi di Beryl Coffee & Chill?")
    
    # Memberi jarak sedikit di bawah judul
    st.write("") 

    # Menggunakan rasio [0.6, 0.4] agar bar chart tetap mungil
    col4, _ = st.columns([0.6, 0.4]) 
    
    with col4:
        # Ukuran mungil (4 x 3) agar teks menu tidak terpotong
        fig4, ax4 = plt.subplots(figsize=(4, 3)) 
        sns.barplot(
            data=top_kombinasi.head(10), 
            x='Frekuensi', 
            y='Kombinasi Menu String', 
            palette='YlOrBr_r', 
            ax=ax4
        )
        
        # Penyesuaian teks sumbu super kecil
        ax4.tick_params(axis='both', labelsize=6)
        ax4.set_xlabel("Jumlah Transaksi", fontsize=7)
        ax4.set_ylabel("", fontsize=7)
        
        plt.tight_layout()
        st.pyplot(fig4)

    # Memberi jarak vertikal sebelum info
    st.markdown("<br>", unsafe_allow_html=True)

    # Insight lengkap sesuai permintaan (menggunakan st.info)
    st.info("""
    **💡 Insight:** Kombinasi menu yang paling sering dibeli pelanggan dalam satu transaksi di Beryl Coffee & Chill adalah **Air Mineral dan Nasi Goreng**. Kombinasi ini memiliki frekuensi transaksi yang jauh lebih tinggi dibandingkan kombinasi lainnya. 

    Selain itu, kombinasi yang melibatkan Air Mineral dengan berbagai menu makanan seperti Ayam Geprek, Florana, dan Kentang Goreng juga termasuk dalam kombinasi dengan frekuensi tinggi. Hal ini menunjukkan bahwa **Air Mineral berperan sebagai produk pelengkap utama** yang sering dibeli bersamaan dengan menu makanan inti. Secara keseluruhan, pola pembelian pelanggan menunjukkan kecenderungan membeli menu makanan bersama minuman pendamping dalam satu transaksi, yang menjadi dasar penting untuk analisis lanjutan seperti *association rule mining* dan strategi *bundling* produk.
    """)
        # Tambahkan satu atau beberapa <br> untuk memberi jarak ke bawah
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
        

# --- TAB 5 & 6 ---
st.subheader(" Kontribusi Menu & Performa Bulanan ")
tab5, tab6 = st.tabs(["🔖 Kontribusi & Tren Produk", "📅 Performa Bulanan"])

with tab5:
    st.markdown(f"### Analisis Performa Produk: {menu_terlaris}")
    
    # Membuat dua kolom untuk visualisasi berdampingan
    col_kiri, col_kanan = st.columns(2)
    
    with col_kiri:
        st.write("##### 5. Menu apa yang memiliki kontribusi penjualan terbesar dan bagaimana perubahan performanya dari waktu ke waktu?")
        fig5a, ax5a = plt.subplots(figsize=(5, 3.5))
        menu_sales.head(10).plot(kind='bar', color='#C28A34', ax=ax5a)
        ax5a.set_ylabel("Total Penjualan (Rp)", fontsize=8)
        ax5a.tick_params(axis='both', labelsize=7)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig5a)
        
    with col_kanan:
        st.write(f"**Tren Penjualan {menu_terlaris}**")
        fig5b, ax5b = plt.subplots(figsize=(5, 3.5))
        ax5b.plot(tren_menu_top.index, tren_menu_top.values, marker='o', markersize=4, color='#2E5A88', linewidth=1.5)
        ax5b.set_ylabel("Total Penjualan (Rp)", fontsize=8)
        ax5b.tick_params(axis='both', labelsize=7)
        ax5b.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig5b)
        
    st.info(f"""**💡 Insight:**  **{menu_terlaris}** menyumbang pendapatan terbesar. 
     dibandingkan menu lainnya. Jika dikaitkan dengan grafik tren penjualan dari waktu ke waktu pada periode September hingga Desember 2025, penjualan Chocolate umumnya menunjukkan pola yang relatif stabil yang mengindikasikan adanya konsistensi pembelian oleh pelanggan. Meskipun terdapat satu lonjakan penjualan yang bersifat ekstrem pada tanggal tertentu, lonjakan tersebut tidak merepresentasikan pola penjualan harian secara umum. Setelahnya, penjualan Chocolate kembali ke tingkat normal. Dengan demikian, kontribusi besar menu Chocolate terhadap total penjualan lebih disebabkan oleh konsistensi performa penjualan sepanjang waktu, bukan oleh lonjakan penjualan sesaat, sehingga Chocolate berperan penting dalam menjaga stabilitas kinerja penjualan Beryl Coffee & Chill""")

with tab6:
    st.markdown("##### 6. Pada periode waktu apa penjualan Beryl Coffee & Chill mencapai puncaknya dan kapan mengalami penurunan?")
    
    # Memberi jarak sedikit di bawah judul
    st.write("") 

    # Menggunakan rasio [0.5, 0.5] agar grafik tetap mungil di tengah
    col6, _ = st.columns([0.5, 0.5]) 
    
    with col6:
        # Ukuran sangat kecil (3.5 x 2.2) agar super kompak
        fig6, ax6 = plt.subplots(figsize=(3.5, 2.2))
        
        penjualan_bulanan.plot(
            marker='o', 
            color='#A67329', 
            linewidth=1, 
            markersize=2.5, 
            ax=ax6
        )
        
        # Penyesuaian font dan grid
        ax6.set_ylabel("Total Penjualan", fontsize=6)
        ax6.tick_params(axis='both', labelsize=5)
        ax6.grid(True, linestyle=':', alpha=0.3)
        
        # Rotasi label bulan diperketat agar tidak bertabrakan
        plt.xticks(rotation=45, ha='right', fontsize=5)
        plt.tight_layout()
        
        st.pyplot(fig6)

    # Memberi jarak vertikal sebelum info
    st.markdown("<br>", unsafe_allow_html=True)

    # Insight Tren Bulanan sesuai permintaan Anda (menggunakan st.info)
    st.info("""
    **💡 Insight:** Berdasarkan grafik tren total penjualan per bulan, penjualan Beryl Coffee & Chill mencapai puncaknya pada bulan **September**. Setelah itu, penjualan mengalami penurunan pada bulan Oktober dan berlanjut hingga bulan November. Meskipun demikian, tingkat penjualan pada bulan November **masih berada di atas bulan Agustus**, yang menunjukkan bahwa penurunan yang terjadi bersifat bertahap dan tidak ekstrem.
    """)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- TAB 7 & 8 ---
st.subheader("Perilaku & Tren Lanjutan")
tab7, tab8 = st.tabs(["🛒 Pola Perilaku", "📉 Moving Average"])

with tab7:
    st.markdown("##### 7. Bagaimana pola perilaku pembelian pelanggan Beryl Coffee & Chill berdasarkan riwayat transaksi yang tersedia?")
    
    # Memberi jarak sedikit di bawah judul
    st.write("") 

    # Menggunakan rasio 50:50 agar visualisasi bersanding mungil
    col7a, col7b = st.columns(2)
    
    with col7a:
        st.write("**Total Belanja per Jenis Order**")
        # Ukuran sangat mungil (3.2 x 2.2)
        fig7a, ax7a = plt.subplots(figsize=(3.2, 2.2))
        sns.barplot(
            data=perilaku_order_jenis.reset_index(), 
            x='Jenis Order', 
            y='Total Belanja', 
            palette='YlOrBr', 
            ax=ax7a
        )
        ax7a.tick_params(axis='both', labelsize=5)
        ax7a.set_xlabel("", fontsize=6)
        ax7a.set_ylabel("Total (Rp)", fontsize=6)
        plt.xticks(rotation=45, ha='right', fontsize=5)
        plt.tight_layout()
        st.pyplot(fig7a)
        
    with col7b:
        st.write("**Distribusi Item per Transaksi**")
        fig7b, ax7b = plt.subplots(figsize=(3.2, 2.2))
        sns.histplot(df['Jumlah_Item_Transaksi'], bins=range(1, 10), color='#A67329', ax=ax7b)
        ax7b.tick_params(axis='both', labelsize=5)
        ax7b.set_xlabel("Jumlah Item", fontsize=6)
        ax7b.set_ylabel("Frekuensi", fontsize=6)
        plt.tight_layout()
        st.pyplot(fig7b)
        
    # Memberi jarak vertikal sebelum info
    st.markdown("<br>", unsafe_allow_html=True)

    # Insight Pola Perilaku sesuai permintaan Anda
    st.info("""
    **💡 Insight:** Berdasarkan riwayat transaksi yang tersedia, pola perilaku pembelian pelanggan Beryl Coffee & Chill didominasi oleh transaksi dengan jenis order **dine-in (table)**, yang memberikan kontribusi belanja paling besar dibandingkan jenis order lainnya. Selain itu, distribusi jumlah item per transaksi menunjukkan bahwa sebagian besar pelanggan membeli antara **satu hingga tiga item** dalam satu transaksi. Pola ini mengindikasikan bahwa pelanggan cenderung melakukan pembelian sederhana dan personal saat berkunjung, dengan fokus pada pengalaman menikmati produk di tempat.
    """)
with tab8:
    st.markdown("##### 8. Bagaimana perkembangan tren penjualan Beryl Coffee & Chill dari waktu ke waktu dan potensi perubahannya di periode mendatang?")
    
    # Memberi jarak sedikit di bawah judul
    st.write("") 

    # Membatasi lebar visualisasi di tengah (50% dari lebar layar)
    col8, _ = st.columns([0.5, 0.5])
    
    with col8:
        # Ukuran sangat kecil (3.5 x 2.2) agar tetap konsisten dan mungil
        fig8, ax8 = plt.subplots(figsize=(3.5, 2.2))
        
        # Plot Penjualan Riil
        ax8.plot(tren_penjualan.index, tren_penjualan['Penjualan'], 
                 label='Riil', marker='o', markersize=2, color='#C28A34', linewidth=1)
        
        # Plot Moving Average
        ax8.plot(tren_penjualan.index, tren_penjualan['MA_3'], 
                 label='MA-3', linestyle='--', color='red', linewidth=1)
        
        # Penyesuaian teks dan estetika grafik
        ax8.tick_params(axis='both', labelsize=5)
        ax8.legend(fontsize=4, loc='upper left')
        ax8.grid(True, alpha=0.2, linestyle=':')
        
        plt.xticks(rotation=45, ha='right', fontsize=5)
        plt.tight_layout()
        st.pyplot(fig8)

    # Memberi jarak vertikal sebelum info
    st.markdown("<br>", unsafe_allow_html=True)

    # Insight Tren Lanjutan sesuai permintaan Anda
    st.info("""
    **💡 Insight:** Berdasarkan grafik tren penjualan dan *moving average* 3 bulan, penjualan Beryl Coffee & Chill sempat meningkat tajam hingga mencapai puncak pada bulan September, kemudian mengalami penurunan pada bulan Oktober dan November. 

    Namun demikian, garis **moving average 3 bulan menunjukkan arah yang cenderung meningkat** pada periode akhir pengamatan. Pola ini mengindikasikan bahwa meskipun terjadi penurunan penjualan dalam jangka pendek, terdapat potensi stabilisasi bahkan pemulihan penjualan pada periode mendatang. Jika tren rata-rata ini berlanjut dan didukung oleh konsistensi perilaku pembelian pelanggan, penjualan Beryl Coffee & Chill diperkirakan tidak akan terus menurun dan berpeluang kembali meningkat atau setidaknya berada pada tingkat yang lebih stabil ke depan.
    """)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- BAGIAN 3: MARKET BASKET ANALYSIS (TABS HORIZONTAL) ---
st.markdown("---")
st.header("🔗 Pola Pembelian (FP-Growth)")

# Persiapan Data
transaction_list = df['Produk'].apply(lambda x: [item.strip() for item in str(x).split(',') if item.strip()]).tolist()
te = TransactionEncoder()
te_array = te.fit(transaction_list).transform(transaction_list)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = fpgrowth(df_trans, min_support=0.01, use_colnames=True).sort_values("support", ascending=False)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules_bi = rules[(rules["confidence"] >= 0.3) & (rules["lift"] >= 1.2) & (rules["consequents"].apply(len) == 1)].sort_values("lift", ascending=False)

# --- TAMPILAN TABS ---
t1, t2, t3, t4 = st.tabs(["📋 Produk Sering Muncul", "🔥 Aturan Asosiasi", "📈 Evaluasi Kualitas", "📦 Rekomendasi Bundling"])

with t1:
    st.subheader("1. Produk yang Sering Muncul Bersamaan")
    display_fi = frequent_itemsets.copy()
    display_fi['itemsets'] = display_fi['itemsets'].apply(lambda x: ', '.join(list(x)))
    st.dataframe(display_fi.head(10), use_container_width=True, hide_index=True)
    st.markdown("""
    > **Cara Baca:** Nilai **Support** menunjukkan seberapa sering kombinasi produk tersebut muncul dalam seluruh transaksi. 
    > Contoh: Support 0.22 berarti item tersebut muncul di 22% dari total transaksi.
    """)

with t2:
    st.subheader("2. Aturan Asosiasi Strategis (Rules Siap Bisnis)")
    rules_display = rules_bi[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
    st.table(rules_display.head(10))
    st.markdown("""
    > **Cara Baca:** Jika pelanggan membeli **Antecedents**, maka kemungkinan besar mereka akan membeli **Consequents**. 
    > Fokus pada nilai **Lift**; jika di atas 1, berarti hubungan antar produk sangat kuat.
    """)

with t3:
    st.subheader("3. Evaluasi Kualitas Aturan")
    cv1, cv2 = st.columns(2)
    with cv1:
        st.write("**Distribusi Nilai Lift**")
        f1, a1 = plt.subplots(figsize=(6,4)); a1.hist(rules_bi["lift"], bins=5, color="#C28A34", edgecolor="white"); st.pyplot(f1)
    with cv2:
        st.write("**Scatter Confidence vs Lift**")
        f2, a2 = plt.subplots(figsize=(6,4)); sns.scatterplot(data=rules_bi, x="confidence", y="lift", color="#A67329", ax=a2); st.pyplot(f2)
    st.markdown("""
    > **Insight Visual:** Semakin tinggi **Confidence** dan **Lift**, semakin valid aturan tersebut untuk dijadikan promo bundling. 
    > Titik-titik di kanan atas adalah peluang emas bisnis.
    """)

with t4:
    st.subheader("4. Rekomendasi Bundling Produk (Aksi Bisnis)")
    if not rules_bi.empty:
        strat = []
        for _, r in rules_bi.iterrows():
            cat = "⭐ Bundling Utama" if r["confidence"] >= 0.45 and r["lift"] >= 1.5 else "📈 Bundling Pendukung"
            strat.append({
                "Produk Utama": ", ".join(list(r["antecedents"])),
                "Rekomendasi": ", ".join(list(r["consequents"])),
                "Tipe Strategi": cat,
                "Aksi": "Buat Paket Hemat" if cat == "⭐ Bundling Utama" else "Suggestive Selling"
            })
        st.dataframe(pd.DataFrame(strat), use_container_width=True, hide_index=True)
    st.markdown("""
    > **Tipe Strategi:** > * **Bundling Utama:** Wajib dibuatkan menu paket karena hubungannya sangat kuat. 
    > * **Bundling Pendukung:** Disarankan untuk ditawarkan oleh kasir (Upselling) saat pelanggan memesan produk utama.
    """)

st.info("💡 **Tips:** Gunakan hasil rekomendasi bundling untuk update menu fisik di Beryl Coffee & Chill.")


# =====================================================
# DATA-DRIVEN DECISION (PUNCAK DASHBOARD) ✨
# =====================================================
st.markdown("---")
st.header("📈 Data-Driven Desicion")

# Logic untuk mengambil insight otomatis
# 1. Ambil rekomendasi bundling terbaik dari FP-Growth (Lift tertinggi)
if not rules_bi.empty:
    best_rule = rules_bi.iloc[0]
    antecedent_name = ", ".join(list(best_rule["antecedents"]))
    consequent_name = ", ".join(list(best_rule["consequents"]))
    bundling_insight = f"Strategi bundling <b>{antecedent_name}</b> dengan <b>{consequent_name}</b> sangat direkomendasikan karena memiliki nilai Lift <b>{best_rule['lift']:.2f}</b> (hubungan sangat kuat)."
else:
    bundling_insight = "Data transaksi saat ini menyarankan fokus pada penjualan item tunggal atau paket hemat menu utama."

# 2. Ambil insight operasional
top_item = top_5_names[0] if top_5_names else "Menu Utama"
busy_day = hari_transaksi.idxmax() if 'hari_transaksi' in locals() else "Hari Sibuk"

# Tampilan UI Keputusan Bisnis
col_strat1, col_strat2, col_strat3 = st.columns(3)

with col_strat1:
    st.markdown(f"""
    <div style="background-color: #fdfcf0; padding: 25px; border-radius: 20px; border-top: 5px solid #C28A34; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <h4 style="color: #A67329; margin-top:0;">🚀 Inventory Priority</h4>
        <p style="font-size: 15px; color: #333; line-height: 1.5;">
            Berdasarkan volume transaksi, stok bahan baku untuk <b>{top_item}</b> harus dijaga 20% lebih banyak untuk menghindari <i>loss of sales</i>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_strat2:
    st.markdown(f"""
    <div style="background-color: #fdfcf0; padding: 25px; border-radius: 20px; border-top: 5px solid #C28A34; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <h4 style="color: #A67329; margin-top:0;">⏰ Workforce Scaling</h4>
        <p style="font-size: 15px; color: #333; line-height: 1.5;">
            Puncak kunjungan terdeteksi pada <b>{peak_hour}</b> hari <b>{busy_day}</b>. Disarankan menambah 1-2 staf paruh waktu untuk menjaga <i>service level agreement</i>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_strat3:
    st.markdown(f"""
    <div style="background-color: #fdfcf0; padding: 25px; border-radius: 20px; border-top: 5px solid #C28A34; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <h4 style="color: #A67329; margin-top:0;">💸 Revenue Engineering</h4>
        <p style="font-size: 15px; color: #333; line-height: 1.5;">
            {bundling_insight} Gunakan teknik <i>suggestive selling</i> pada kasir untuk menawarkan menu tambahan tersebut.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Kesimpulan Akhir yang dinamis
st.info(f"💡 **Data Insight:** Beryl Coffee & Chill menunjukkan performa stabil. Fokus pada pelanggan **Dine-in** dan optimalisasi menu **{top_item}** adalah kunci pertumbuhan profit periode mendatang.")