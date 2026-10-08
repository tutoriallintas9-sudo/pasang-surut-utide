# ============================================================
# STREAMLIT APP ANALISA PASANG SURUT - UTIDE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utide import solve, reconstruct
import math
from sklearn.metrics import r2_score
from datetime import datetime
import io
import zipfile


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Analisa Pasang Surut",
    layout="wide"
)


st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #EAF7FF 0%,
            #FFFFFF 50%,
            #FFF8D6 100%
        );
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    .judul-aplikasi {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(
            90deg,
            #0066B3 0%,
            #0088CC 25%,
            #FFD900 65%,
            #F5A000 75%
        );
        
        background-size: 200% auto;
        
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: kilau 10s linear infinite;
        margin-bottom: 20px;
    }
    
    @keyframes kilau {
        0% {
            background-position: 200% center;
        }

        100% {
            background-position: -200% center;
        }
    }
    
    </style>

    <div class="judul-aplikasi">
        Aplikasi Analisa Pasang Surut
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 1. UNGGAH DATA
# ============================================================

col_main, col_info = st.columns(
    [3.5, 1],
    gap="large"
)

# ============================================================
# KOLOM KIRI
# ============================================================

with col_main:


    st.header("1. Unggah Data Pasang Surut")

    st.markdown("""
    **Format file Excel yang diupload harus memiliki minimal 2 kolom dengan nama:**

    - `Tanggal` : Tanggal dan waktu pengamatan
    - `Elevasi` : Tinggi muka air (meter)
    """)

    uploaded_file = st.file_uploader(
        "Pilih file Excel",
        type=["xlsx"]
    )


    # ============================================================
    # INPUT LATITUDE
    # ============================================================

    latitude = st.number_input(
        "Masukkan nilai latitude lokasi pengamatan (dalam derajat desimal):",
        min_value=-90.0,
        max_value=90.0,
        value=0.000000,
        format="%.6f"
    )
    st.markdown(
        "<small><span style='color:red;'>* Latitude wajib diinput untuk faktor koreksi nodal UTide</span></small>",
        unsafe_allow_html=True
    )

    # ============================================================
    # PERIODE PREDIKSI
    # ============================================================

    st.markdown("### Periode dan Interval Prediksi Pasang Surut")

    start_pred = st.date_input(
        "Tanggal Mulai",
        datetime(2025, 1, 1)
    )

    end_pred = st.date_input(
        "Tanggal Akhir",
        datetime(2025, 6, 30)
    )


    # ============================================================
    # INTERVAL PREDIKSI
    # ============================================================

    interval_options = {
        "6 Jam": "6h",
        "3 Jam": "3h",
        "1 Jam": "1h",
        "30 Menit": "30min",
        "15 Menit": "15min"
    }

    interval_label = st.selectbox(
        "Pilih Interval Prediksi",
        list(interval_options.keys()),
        index=1
    )

    interval = interval_options[interval_label]


    # ============================================================
    # VALIDASI TANGGAL
    # ============================================================

    if start_pred >= end_pred:
        st.error(
            "Tanggal mulai harus lebih awal dari tanggal akhir."
        )


    # ============================================================
    # TOMBOL ANALISA
    # ============================================================

    run_analysis = st.button(
        "🔍 Analisa Pasang Surut",
        type="primary"
    )


# ============================================================
# KOLOM KANAN - KETERANGAN
# ============================================================

with col_info:

    st.markdown(
        """
        <div class="kotak-keterangan">

        <h3>ℹ️ Keterangan</h3>

        <p>
        Aplikasi ini digunakan untuk melakukan analisis
        pasang surut menggunakan metode harmonik UTide.
        </p>

        <p>
        <b>Data yang diperlukan:</b><br>
        • Tanggal<br>
        • Elevasi<br>
        • Latitude lokasi pengamatan
        </p>

        <p>
        <b>Parameter analisis:</b><br>
        • Periode prediksi<br>
        • Interval prediksi<br>
        • Faktor koreksi nodal
        </p>

        <p>
        Hasil analisis meliputi komponen harmonik,
        Formzahl, klasifikasi pasang surut,
        prediksi, RMSE, R², MSL, HWS, dan LWS.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Contoh Format Excel")

    contoh_data = pd.DataFrame({
        "Tanggal": [
            "01-07-25 0:00",
            "01-07-25 1:00",
            "01-07-25 2:00",
            "01-07-25 3:00",
            "01-07-25 4:00"
        ],
        "Elevasi": [
            140,
            136,
            132,
            128,
            126
        ]
    })

    st.dataframe(
        contoh_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Nama kolom harus: Tanggal dan Elevasi."
    )


# ============================================================
# PROSES ANALISIS
# ============================================================

if uploaded_file is not None and run_analysis:

    # ========================================================
    # VALIDASI LATITUDE
    # ========================================================

    if latitude == 0.000000:

        st.warning(
            "⚠️ Latitude wajib diinput untuk faktor koreksi nodal UTide."
        )

        st.stop()

    # ========================================================
    # BACA DATA EXCEL
    # ========================================================

    try:

        df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(
            f"File Excel tidak dapat dibaca: {e}"
        )

        st.stop()


    # ========================================================
    # VALIDASI KOLOM
    # ========================================================

    required_columns = ["Tanggal", "Elevasi"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}"
        )

        st.stop()


    # ========================================================
    # COPY DATA
    # ========================================================

    bfill_imputed = df.copy(deep=True)


    # ========================================================
    # KONVERSI DATA
    # ========================================================

    bfill_imputed["Tanggal"] = pd.to_datetime(
        bfill_imputed["Tanggal"],
        errors="coerce"
    )

    bfill_imputed["Elevasi"] = pd.to_numeric(
        bfill_imputed["Elevasi"],
        errors="coerce"
    )


    # ========================================================
    # HAPUS DATA TIDAK VALID
    # ========================================================

    bfill_imputed = bfill_imputed.dropna(
        subset=["Tanggal", "Elevasi"]
    )


    # ========================================================
    # CEK JUMLAH DATA
    # ========================================================

    if len(bfill_imputed) < 10:

        st.error(
            "Data pengamatan terlalu sedikit untuk dilakukan analisis UTide."
        )

        st.stop()


    # ========================================================
    # DATA UNTUK ANALISIS
    # ========================================================

    tanggal = bfill_imputed["Tanggal"]

    sensor = np.asarray(
        bfill_imputed["Elevasi"],
        dtype=float
    )


    # ========================================================
    # 2. GRAFIK OBSERVASI
    # ========================================================

    st.header("2. Grafik Pasang Surut Observasi")

    fig_obs, ax_obs = plt.subplots(
        figsize=(18, 8)
    )

    ax_obs.plot(
        tanggal,
        sensor,
        color="black",
        linewidth=1
    )

    ax_obs.set_xlabel(
        "Tanggal"
    )

    ax_obs.set_ylabel(
        "Tinggi Air [m]"
    )

    ax_obs.set_title(
        "Grafik Pasang Surut Observasi"
    )

    ax_obs.grid()

    fig_obs.tight_layout()

    st.pyplot(fig_obs)


    # ========================================================
    # 3. ANALISIS HARMONIK UTIDE
    # ========================================================

    constituents = [
        "M2",
        "S2",
        "K1",
        "O1",
        "P1",
        "K2",
        "N2",
        "M4",
        "MS4"
    ]


    try:

        decompose_utide = solve(
            tanggal,
            sensor,
            lat=latitude,
            constit=constituents,
            nodal=True,
            trend=True,
            method="ols",
            conf_int="linear"
        )

    except Exception as e:

        st.error(
            f"Analisis UTide gagal dilakukan: {e}"
        )

        st.stop()


    # ========================================================
    # TABEL KOMPONEN HARMONIK
    # ========================================================

    DatFrame_UTide = pd.DataFrame({
        "Name": decompose_utide["name"],
        "Freq [cph]": decompose_utide["aux"]["frq"],
        "Amplitude [m]": decompose_utide["A"],
        "Amp CI [m]": decompose_utide["A_ci"],
        "Phase [°]": decompose_utide["g"],
        "Phase CI [°]": decompose_utide["g_ci"]
    })


    st.header("3. Tabel Komponen Harmonik")

    st.dataframe(
        DatFrame_UTide.style.format({
            "Freq [cph]": "{:.4f}",
            "Amplitude [m]": "{:.4f}",
            "Amp CI [m]": "{:.4f}",
            "Phase [°]": "{:.4f}",
            "Phase CI [°]": "{:.4f}"
        }),
        use_container_width=True
    )


    # ========================================================
    # 4. PERHITUNGAN FORMZAHL
    # ========================================================

    def get_amplitude(name):

        idx = np.where(
            decompose_utide["name"] == name
        )[0]

        if len(idx) > 0:
            return float(
                decompose_utide["A"][idx[0]]
            )

        return 0.0


    M2 = get_amplitude("M2")
    S2 = get_amplitude("S2")
    K1 = get_amplitude("K1")
    O1 = get_amplitude("O1")


    # ========================================================
    # FORMZAHL
    # ========================================================

    denominator = M2 + S2

    if denominator != 0:

        Formzahl = (K1 + O1) / denominator

    else:

        Formzahl = 0.0


    # ========================================================
    # KLASIFIKASI PASANG SURUT
    # ========================================================

    if Formzahl > 3.0:

        jenis_pasang_surut = (
            "Pasang Surut Harian Tunggal (Diurnal)"
        )

    elif 1.5 < Formzahl <= 3.0:

        jenis_pasang_surut = (
            "Pasang Surut Campuran "
            "Condong ke Harian Tunggal"
        )

    elif 0.25 < Formzahl <= 1.5:

        jenis_pasang_surut = (
            "Pasang Surut Campuran "
            "Condong ke Harian Ganda"
        )

    else:

        jenis_pasang_surut = (
            "Pasang Surut Harian Ganda (Semidiurnal)"
        )


    # ========================================================
    # GRAFIK FORMZAHL
    # ========================================================

    fig_formzahl, ax_formzahl = plt.subplots(
        figsize=(6, 4)
    )

    ax_formzahl.bar(
        ["Formzahl"],
        [Formzahl],
        color="orange"
    )

    ax_formzahl.set_ylim(
        0,
        max(Formzahl + 1, 4)
    )

    ax_formzahl.axhline(
        3.0,
        color="red",
        linestyle="--",
        label="Diurnal"
    )

    ax_formzahl.axhline(
        1.5,
        color="green",
        linestyle="--",
        label="Mixed-Diurnal"
    )

    ax_formzahl.axhline(
        0.25,
        color="blue",
        linestyle="--",
        label="Mixed-Semidiurnal"
    )

    ax_formzahl.set_ylabel(
        "Nilai Formzahl"
    )

    ax_formzahl.set_title(
        "Formzahl"
    )

    ax_formzahl.legend()

    ax_formzahl.text(
        0,
        Formzahl + 0.1,
        f"{Formzahl:.4f}",
        ha="center",
        color="blue",
        fontsize=12
    )

    ax_formzahl.grid(
        axis="y",
        alpha=0.3
    )

    fig_formzahl.tight_layout()


    # ========================================================
    # TAMPILKAN FORMZAHL
    # ========================================================

    st.header(
        "4. Visualisasi Formzahl dan Klasifikasi Pasang Surut"
    )

    col_grafik, col_spacer, col_keterangan = st.columns([0.8,  0.2, 1])

    # =========================
    # GRAFIK
    # =========================

    with col_grafik:

        st.pyplot(
            fig_formzahl,
            use_container_width=True
        )


    # =========================
    # KETERANGAN
    # =========================

    with col_keterangan:

        st.markdown(
            f"### Hasil Analisis"
        )

        st.markdown(
            f"**Nilai Formzahl:** {Formzahl:.4f}"
        )

        st.markdown(
            f"**Jenis Pasang Surut:** {jenis_pasang_surut}"
        )

        st.markdown("""
        ### Kategori Formzahl

        - **F ≤ 0.25** : Pasang surut harian ganda *(Semidiurnal)*
        - **0.25 < F ≤ 1.50** : Campuran condong ke harian ganda
        - **1.50 < F ≤ 3.00** : Campuran condong ke harian tunggal
        - **F > 3.00** : Pasang surut harian tunggal *(Diurnal)*
        """)


    # ========================================================
    # 5. PREDIKSI PASANG SURUT
    # ========================================================

    st.header(
        "5. Prediksi Pasang Surut"
    )


    timepred_UTIDE = pd.date_range(
        start=start_pred,
        end=end_pred,
        freq=interval
    )


    if len(timepred_UTIDE) == 0:

        st.error(
            "Periode prediksi tidak menghasilkan tanggal."
        )

        st.stop()


    try:

        tidepred_UTIDE = reconstruct(
            timepred_UTIDE,
            decompose_utide,
            verbose=True
        )

    except Exception as e:

        st.error(
            f"Prediksi UTide gagal dilakukan: {e}"
        )

        st.stop()


    h_out_predutide = np.asarray(
        tidepred_UTIDE.h,
        dtype=float
    )


    # ========================================================
    # STATISTIK PREDIKSI
    # ========================================================

    MSL_rec = np.mean(
        h_out_predutide
    )

    HWS_rec = np.max(
        h_out_predutide
    )

    LWS_rec = np.min(
        h_out_predutide
    )

    # Index waktu HWS dan LWS
    idx_HWS = np.argmax(h_out_predutide)
    idx_LWS = np.argmin(h_out_predutide)
    
    # Waktu terjadinya HWS dan LWS
    waktu_HWS = timepred_UTIDE[idx_HWS]
    waktu_LWS = timepred_UTIDE[idx_LWS]
    
    # Tunggang Pasang Prediksi
    Tunggang_Pasang_rec = HWS_rec - LWS_rec

    # ========================================================
    # GRAFIK PREDIKSI
    # ========================================================

    fig_pred, ax_pred = plt.subplots(
        figsize=(17, 6)
    )

    ax_pred.plot(
        timepred_UTIDE,
        h_out_predutide,
        color="blue",
        linewidth=1
    )

    ax_pred.set_xlabel(
        "Tanggal"
    )

    ax_pred.set_ylabel(
        "Tinggi Air [m]"
    )

    ax_pred.set_title(
        "Prediksi Tinggi Pasang Surut"
    )

    ax_pred.grid()

    fig_pred.tight_layout()

    st.pyplot(
        fig_pred
    )


    # ========================================================
    # DATA PREDIKSI
    # ========================================================

    df_prediksi = pd.DataFrame({
        "Tanggal": timepred_UTIDE,
        "Elevasi": h_out_predutide
    })


    # ========================================================
    # REKONSTRUKSI DATA OBSERVASI
    # ========================================================

    try:

        tide_utide = reconstruct(
            tanggal,
            decompose_utide,
            verbose=False
        )

        pred_utide = np.asarray(
            tide_utide.h,
            dtype=float
        )

    except Exception as e:

        st.error(
            f"Rekonstruksi data observasi gagal: {e}"
        )

        st.stop()


    # ========================================================
    # RESIDUAL
    # ========================================================

    residual_utide = (
        sensor - pred_utide
    )


    # ========================================================
    # RMSE
    # ========================================================

    RMSE_UTIDE = math.sqrt(
        np.mean(
            np.square(
                sensor - pred_utide
            )
        )
    )


    # ========================================================
    # R-SQUARE
    # ========================================================

    R_square = r2_score(
        sensor,
        pred_utide
    )


    # ========================================================
    # GRAFIK OBSERVASI VS PREDIKSI
    # ========================================================

    fig_obs_pred, ax_obs_pred = plt.subplots(
        figsize=(10, 4)
    )

    ax_obs_pred.plot(
        tanggal,
        sensor,
        label="Observasi",
        color="black"
    )

    ax_obs_pred.plot(
        tanggal,
        pred_utide,
        label="Prediksi",
        color="red"
    )

    ax_obs_pred.set_title(
        "Observasi vs Prediksi"
    )

    ax_obs_pred.set_xlabel(
        "Tanggal"
    )

    ax_obs_pred.set_ylabel(
        "Tinggi Air [m]"
    )

    ax_obs_pred.legend()

    ax_obs_pred.grid(
        True
    )

    fig_obs_pred.tight_layout()


    # ========================================================
    # GRAFIK RESIDUAL
    # ========================================================

    fig_residual, ax_residual = plt.subplots(
        figsize=(10, 4)
    )

    ax_residual.plot(
        tanggal,
        residual_utide,
        label="Residual",
        color="green"
    )

    ax_residual.set_title(
        "Residual"
    )

    ax_residual.set_xlabel(
        "Tanggal"
    )

    ax_residual.set_ylabel(
        "Selisih [m]"
    )

    ax_residual.legend()

    ax_residual.grid(
        True
    )

    fig_residual.tight_layout()


    # ========================================================
    # TAMPILKAN GRAFIK
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.pyplot(
            fig_obs_pred
        )

    with col2:

        st.pyplot(
            fig_residual
        )


    # ========================================================
    # 6. RINGKASAN ANALISIS
    # ========================================================

    st.header(
        "6. Ringkasan Analisis"
    )

    st.markdown(f"""
    - **Formzahl**: {Formzahl:.2f}  
    - **Jenis Pasang Surut**: {jenis_pasang_surut}  
    - **RMSE**: {RMSE_UTIDE:.2f}  
    - **R²**: {R_square:.2f}  
    - **MSL Prediksi**: {MSL_rec:.2f} m
    - **HWS Prediksi**: {HWS_rec:.2f} m 
    - **Waktu HWS**: {waktu_HWS.strftime("%d-%m-%Y %H:%M")}
    - **LWS Prediksi**: {LWS_rec:.2f} m 
    - **Waktu LWS**: {waktu_LWS.strftime("%d-%m-%Y %H:%M")} 
    - **Tunggang Pasang**: {Tunggang_Pasang_rec:.2f} m  
    """)


    # ========================================================
    # 7. EKSPOR HASIL
    # ========================================================

    st.header(
        "7. Ekspor Hasil"
    )


    # ========================================================
    # FUNGSI GRAFIK KE BYTES
    # ========================================================

    def fig_to_bytes(fig):

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="jpg",
            dpi=200,
            bbox_inches="tight"
        )

        buf.seek(0)

        return buf.read()


    # ========================================================
    # ZIP SEMUA GRAFIK
    # ========================================================

    with io.BytesIO() as buffer:

        with zipfile.ZipFile(
            buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zipf:

            zipf.writestr(
                "Grafik_Observasi.jpg",
                fig_to_bytes(fig_obs)
            )

            zipf.writestr(
                "Grafik_Formzahl.jpg",
                fig_to_bytes(fig_formzahl)
            )

            zipf.writestr(
                "Grafik_Observasi_vs_Prediksi.jpg",
                fig_to_bytes(fig_obs_pred)
            )

            zipf.writestr(
                "Grafik_Residual.jpg",
                fig_to_bytes(fig_residual)
            )

            zipf.writestr(
                "Grafik_Prediksi.jpg",
                fig_to_bytes(fig_pred)
            )

        buffer.seek(0)

        st.download_button(
            "📈 Unduh Semua Grafik JPG (ZIP)",
            data=buffer,
            file_name="Grafik_Pasang_Surut.zip",
            mime="application/zip"
        )


    # ========================================================
    # DOWNLOAD KOMPONEN HARMONIK
    # ========================================================

    st.download_button(
        "📄 Unduh Komponen Harmonik",
        data=DatFrame_UTide.to_csv(
            index=False
        ),
        file_name="Komponen_Harmonik.csv",
        mime="text/csv"
    )


    # ========================================================
    # DOWNLOAD DATA PREDIKSI
    # ========================================================

    st.download_button(
        "📅 Unduh Data Prediksi",
        data=df_prediksi.to_csv(
            index=False
        ),
        file_name="Prediksi_Pasang.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:34px; font-weight:bold;">
        BY :  
        <span style="
            background: linear-gradient(
                90deg,
                #0066B3 0%,
                #008FD5 35%,
                #FFD400 65%,
                #F5A400 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">
            SEGARAGIS
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
