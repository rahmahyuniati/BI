import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from base64 import b64encode
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

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

# --- 4. TAMPILAN HEADER & KPI ---
st.markdown(f"""
<style>
    .header-card {{ background: linear-gradient(135deg, #C28A34, #A67329); padding: 40px 50px; border-radius: 25px; margin-bottom: 30px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); }}
    .header-top-row {{ display: flex; align-items: center; gap: 25px; margin-bottom: 10px; }}
    .logo-img {{ width: 80px; height: 80px; object-fit: cover; border-radius: 50%; border: 3px solid white; }}
    .header-title {{ font-size: 42px; font-weight: 800; color: white !important; margin: 0; letter-spacing: -1px; }}
    .header-subtitle {{ color: #FDFCF0; font-size: 18px; font-weight: 400; opacity: 0.95; margin-bottom: 20px; }}
    .badge {{ background-color: rgba(255,255,255,0.2); color: white; padding: 6px 16px; border-radius: 50px; font-size: 13px; font-weight: 600; margin-right: 8px; border: 1px solid rgba(255,255,255,0.3); }}
    .kpi-wrapper {{ display: flex; gap: 20px; margin-bottom: 30px; }}
    .beryl-card {{ flex: 1; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-bottom: 6px solid #C28A34; }}
    .kpi-title {{ font-size: 11px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
    .kpi-value {{ font-size: 24px; font-weight: 850; color: #222; margin-top: 10px; }}
</style>

<div class="header-card">
    <div class="header-top-row">
        <img src="data:image/jpeg;base64,{logo_base64}" class="logo-img">
        <h1 class="header-title"> Beryl Coffee & Chill</h1>
    </div>
    <div class="header-subtitle">Analisis Pola Pembelian & Prediksi Tren Penjualan</div>
    <div><span class="badge">EDA</span><span class="badge">FP-Growth</span><span class="badge">LSTM</span></div>
</div>

<div class="kpi-wrapper">
    <div class="beryl-card"><div class="kpi-title">💰 Total Revenue</div><div class="kpi-value">Rp {total_revenue/1e6:.2f}M</div></div>
    <div class="beryl-card"><div class="kpi-title">👥 Traffic</div><div class="kpi-value">{total_trans:,}</div></div>
    <div class="beryl-card"><div class="kpi-title">🏆 Champion</div><div class="kpi-value" style="font-size: 18px;">{top_prod}</div></div>
    <div class="beryl-card"><div class="kpi-title">🛒 Basket Size</div><div class="kpi-value">Rp {avg_trans:,.0f}</div></div>
    <div class="beryl-card"><div class="kpi-title">⚡ Prime Time</div><div class="kpi-value">{peak_hour}</div></div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# EDA - ANALISIS POLA WAKTU (JAM & HARI) - FIXED
# =====================================================
st.header("⏰ Analisis Pola Waktu Operasional")

# 1) Analisis Berdasarkan Jam
col_jam, col_hari = st.columns(2)

with col_jam:
    st.subheader("1. Transaksi Berdasarkan Jam")
    jam_transaksi = df['Jam_Int'].value_counts().sort_index()
    
    fig_jam, ax_jam = plt.subplots(figsize=(10, 5))
    ax_jam.plot(jam_transaksi.index, jam_transaksi.values, marker="o", linestyle='-', color='#1f77b4', linewidth=2)
    ax_jam.fill_between(jam_transaksi.index, jam_transaksi.values, alpha=0.1, color='#1f77b4')
    ax_jam.set_title("Jumlah Transaksi Berdasarkan Jam", fontsize=12, fontweight='bold')
    ax_jam.set_xlabel("Jam", fontsize=10)
    ax_jam.set_ylabel("Jumlah Transaksi", fontsize=10)
    ax_jam.set_xticks(range(0, 24))
    ax_jam.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_jam)
    st.info(f"Puncak transaksi pada jam: **{jam_transaksi.idxmax()}:00**")

with col_hari:
    # 2) Analisis Berdasarkan Hari
    st.subheader("2. Transaksi Berdasarkan Hari")
    df['Nama Hari'] = df['Tanggal Order'].dt.day_name()
    urutan_hari = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Nama Hari'] = pd.Categorical(df['Nama Hari'], categories=urutan_hari, ordered=True)
    hari_transaksi = df['Nama Hari'].value_counts().sort_index()

    fig_hari, ax_hari = plt.subplots(figsize=(10, 5))
    sns.barplot(x=hari_transaksi.index, y=hari_transaksi.values, palette='YlOrBr', ax=ax_hari)
    ax_hari.set_title("Jumlah Transaksi Berdasarkan Hari", fontsize=12, fontweight='bold')
    ax_hari.set_xlabel("Hari", fontsize=10)
    ax_hari.set_ylabel("Jumlah Transaksi", fontsize=10)
    ax_hari.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig_hari)
    st.info(f"Puncak transaksi pada hari: **{hari_transaksi.idxmax()}**")

# --- BAGIAN 1: EDA JENIS ORDER ---
st.markdown("---")
st.subheader("📊 Total Belanja per Jenis Order")
perilaku_order_jenis = df.groupby('Jenis Order')['Total Penjualan (Rp)'].sum().reset_index().sort_values(by='Total Penjualan (Rp)', ascending=False)
c_order, c_ins_order = st.columns([2, 1])
with c_order:
    st.bar_chart(data=perilaku_order_jenis, x='Jenis Order', y='Total Penjualan (Rp)', color='#C28A34')
with c_ins_order:
    st.write("### 💡 Insight")
    st.info("Penjualan terbesar berasal dari **Dine-in (Table)**. Fokuskan pelayanan meja yang prima.")
    st.dataframe(perilaku_order_jenis, hide_index=True)

# --- BAGIAN 2: EDA DISTRIBUSI ITEM ---
st.subheader("🛒 Distribusi Jumlah Item per Transaksi")
df['Jumlah_Item_Transaksi'] = df['Produk'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
c_dist, c_ins_dist = st.columns([2, 1])
with c_dist:
    item_dist = df['Jumlah_Item_Transaksi'].value_counts().sort_index().reset_index()
    item_dist.columns = ['Jumlah Item', 'Frekuensi']
    st.bar_chart(data=item_dist, x='Jumlah Item', y='Frekuensi', color='#A67329')
with c_ins_dist:
    st.write("### 💡 Insight")
    st.warning("Rata-rata pelanggan membeli **1-3 item**. Strategi bundling sangat diperlukan untuk meningkatkan jumlah item per struk.")
    st.metric("Rata-rata Item/Transaksi", f"{df['Jumlah_Item_Transaksi'].mean():.1f} Produk")


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
# 🤖 LSTM – PREDIKSI TREN PENJUALAN
# =====================================================
st.markdown("---")
st.header("🤖 Prediksi Tren Penjualan (LSTM)")

# ===============================
# 1. AGREGASI PENJUALAN HARIAN
# ===============================
daily_sales = (
    df
    .groupby("Tanggal Order")["Total Penjualan (Rp)"]
    .sum()
    .reset_index()
    .sort_values("Tanggal Order")
)

st.subheader("📊 Data Penjualan Harian")
st.dataframe(daily_sales.tail(10), use_container_width=True)

# ===============================
# 2. CEK DATA CUKUP UNTUK LSTM
# ===============================
if len(daily_sales) < 14:
    st.warning("⚠️ Data terlalu sedikit untuk LSTM (minimal 14 hari)")
else:
    # ===============================
    # 3. NORMALISASI DATA
    # ===============================
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(
        daily_sales["Total Penjualan (Rp)"].values.reshape(-1, 1)
    )

    # ===============================
    # 4. BUAT SEQUENCE DATA
    # ===============================
    def create_sequence(data, window=7):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i+window])
            y.append(data[i+window])
        return np.array(X), np.array(y)

    WINDOW = 7
    X, y = create_sequence(sales_scaled, WINDOW)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # ===============================
    # 5. MODEL LSTM
    # ===============================
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(WINDOW, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # ===============================
    # 6. TRAINING
    # ===============================
    with st.spinner("⏳ Training model LSTM..."):
        model.fit(X, y, epochs=20, batch_size=4, verbose=0)

    # ===============================
    # 7. PREDIKSI DATA HISTORIS
    # ===============================
    pred_scaled = model.predict(X)
    pred_actual = scaler.inverse_transform(pred_scaled)

    actual_plot = daily_sales["Total Penjualan (Rp)"].values[WINDOW:]
    date_plot = daily_sales["Tanggal Order"].values[WINDOW:]

    # ===============================
    # 8. VISUALISASI
    # ===============================
    st.subheader("📈 Tren Penjualan: Aktual vs Prediksi")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_sales["Tanggal Order"], daily_sales["Total Penjualan (Rp)"],
            label="Aktual", linewidth=2, alpha=0.4)
    ax.plot(date_plot, pred_actual,
            label="Prediksi LSTM", linewidth=2)

    ax.set_title("Prediksi Tren Penjualan Harian Beryl Coffee & Chill")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Total Penjualan (Rp)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

    # ===============================
    # 9. PREDIKSI 1 HARI KE DEPAN
    # ===============================
    last_seq = sales_scaled[-WINDOW:].reshape(1, WINDOW, 1)
    next_day_scaled = model.predict(last_seq)
    next_day_sales = scaler.inverse_transform(next_day_scaled)[0][0]

    st.metric(
        "📌 Estimasi Penjualan Hari Berikutnya",
        f"Rp {next_day_sales:,.0f}"
    )

    # ===============================
    # 10. INSIGHT OTOMATIS
    # ===============================
    avg_sales = daily_sales["Total Penjualan (Rp)"].mean()

    if next_day_sales > avg_sales:
        st.success("📈 Insight: Penjualan diprediksi **MENINGKAT** dibandingkan rata-rata.")
    else:
        st.info("📉 Insight: Penjualan diprediksi **RELATIF STABIL / MENURUN**.")
