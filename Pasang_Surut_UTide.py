# Streamlit App Analisa Pasang Surut dengan Prediksi Tanggal di Bawah Latitude

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

st.set_page_config(page_title="Aplikasi Analisa Pasang Surut", page_icon="🌊")
st.image(
    "https://blogger.googleusercontent.com/img/a/AVvXsEgNSHUPAWW3jVgULWuvcgEDRIWYFS6P22VHyAOzGzCoRyFFVWau9sl3bZlumjahc0c6foKUvWfPpYiCHugPw9dric5xp8X92nnrqhoJuwqiitGdOEC-BOKte2mu3KnvnEr9TZLb7uvEYJPNLZYpLCBpqeblJlU-jLZgMn4n59LXlBfan3N93VJEBrNjAHQ=s1600",
    width=300,   # 🔹 ubah angka ini sesuai kebutuhan
)
st.title("Aplikasi Analisa Pasang Surut")

st.header("1. Upload Data")
st.markdown("""
**Format file Excel yang diupload harus memiliki minimal 2 kolom dengan nama:**

- `Tanggal`: Tanggal dan waktu
- `Elevasi`: Tinggi air (meter)
""")

uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

latitude = st.number_input("Masukkan nilai latitude lokasi pengamatan (dalam derajat desimal):", 
                           min_value=-90.0, max_value=90.0, value=0.000000, format="%.6f")

st.markdown("### Pilih Rentang Prediksi Pasang Surut")
start_pred = st.date_input("Tanggal Mulai", datetime(2025, 1, 1))
end_pred = st.date_input("Tanggal Akhir", datetime(2025, 6, 30))

interval_options = {
    "6 Jam": "6H",
    "3 Jam": "3H",
    "1 Jam": "1H",
    "30 Menit": "30min",
    "15 Menit": "15min"
}
interval_label = st.selectbox("Pilih Interval Prediksi", list(interval_options.keys()), index=1)
interval = interval_options[interval_label]


if start_pred >= end_pred:
    st.error("Tanggal mulai harus lebih awal dari tanggal akhir.")

run_analysis = st.button("Analisa Pasang Surut")

if uploaded_file is not None and run_analysis:
    df = pd.read_excel(uploaded_file)
    bfill_imputed = df.copy(deep=True)
    bfill_imputed.fillna(method='bfill', inplace=True)
    tanggal = pd.to_datetime(bfill_imputed['Tanggal'])
    sensor = np.array(bfill_imputed['Elevasi'])

    st.header("2. Grafik Pasang Surut Observasi")
    fig_obs, ax_obs = plt.subplots(figsize=(17, 8))
    ax_obs.plot(tanggal, sensor, color='black', linewidth=1)
    ax_obs.set(xlabel='Tanggal', ylabel='Tinggi Air [m]', title='Grafik Pasang Surut')
    ax_obs.grid()
    st.pyplot(fig_obs)

    constituents = ['M2', 'S2', 'K1', 'O1', 'P1', 'K2', 'N2', 'M4', 'MS4']
    decompose_utide = solve(tanggal, sensor, lat=latitude,
                            constit=constituents, nodal=False,
                            trend=False, method='ols', conf_int='linear')

    DatFrame_UTide = pd.DataFrame({
        'Name': decompose_utide['name'],
        'Freq [cph]': decompose_utide['aux']['frq'],
        'Amplitude [m]': decompose_utide['A'],
        'Amp CI [m]': decompose_utide['A_ci'],
        'Phase [°]': decompose_utide['g'],
        'Phase CI [°]': decompose_utide['g_ci']
    })

    st.header("3. Tabel Komponen Harmonik")
    st.dataframe(DatFrame_UTide.style.format({
        'Freq [cph]': '{:.4f}',
        'Amplitude [m]': '{:.4f}',
        'Amp CI [m]': '{:.4f}',
        'Phase [°]': '{:.4f}',
        'Phase CI [°]': '{:.4f}'
    }))

    MSL = np.mean(sensor)
    amp = lambda x: decompose_utide['A'][decompose_utide['name'] == x][0] if x in decompose_utide['name'] else 0
    M2, S2, K1, O1, P1, K2, M4, MS4 = [amp(x) for x in ['M2','S2','K1','O1','P1','K2','M4','MS4']]

    HWS = MSL + M2 + S2 + K1 + O1 + P1 + K2 + M4 + MS4
    LWS = MSL - (M2 + S2 + K1 + O1 + P1 + K2 + M4 + MS4)
    MHWS = MSL + M2 + S2
    MLWS = MSL - (M2 + S2)
    MHWL = MSL + M2 + K1 + O1
    MLWL = MSL - (M2 + K1 + O1)
    HHWL = MSL + M2 + S2 + K1 + O1 + P1 + K2
    LLWL = MSL - (M2 + S2 + K1 + O1 + P1 + K2)
    Tunggang_Pasang = HWS - LWS
    Formzahl = (K1 + O1) / (M2 + S2) if (M2 + S2) != 0 else 0

    if Formzahl > 3.0:
        jenis_pasang_surut = "Pasang Surut Harian Tunggal (Diurnal)"
    elif 1.5 < Formzahl <= 3.0:
        jenis_pasang_surut = "Pasang Surut Campuran Condong ke Harian Tunggal"
    elif 0.25 < Formzahl <= 1.5:
        jenis_pasang_surut = "Pasang Surut Campuran Condong ke Harian Ganda"
    else:
        jenis_pasang_surut = "Pasang Surut Harian Ganda (Semidiurnal)"

    df_elevasi = pd.DataFrame({
        'Parameter': [
            'MSL', 'HWS', 'LWS', 'MHWS', 'MLWS',
            'MHWL', 'MLWL', 'HHWL', 'LLWL', 'Tunggang Pasang',
            'Formzahl', 'Jenis Pasut'
        ],
        'Elevasi (m)': [
            MSL, HWS, LWS, MHWS, MLWS,
            MHWL, MLWL, HHWL, LLWL, Tunggang_Pasang,
            Formzahl, jenis_pasang_surut
        ]
    })

    fig_formzahl, ax_formzahl = plt.subplots(figsize=(6, 4))
    ax_formzahl.bar(["Formzahl"], [Formzahl], color='orange')
    ax_formzahl.set_ylim(0, max(Formzahl + 1, 4))
    ax_formzahl.axhline(3.0, color='red', linestyle='--', label='Diurnal')
    ax_formzahl.axhline(1.5, color='green', linestyle='--', label='Mixed-Diurnal')
    ax_formzahl.axhline(0.25, color='blue', linestyle='--', label='Mixed-Semidiurnal')
    ax_formzahl.set_ylabel("Nilai Formzahl")
    ax_formzahl.legend()
    ax_formzahl.text(0, Formzahl + 0.1, f"{Formzahl:.4f}", ha='center', color='blue', fontsize=12)

    st.header("4. Visualisasi Formzahl dan Klasifikasi Pasang Surut")
    st.pyplot(fig_formzahl)
    st.markdown(f"**Jenis Pasang Surut:** {jenis_pasang_surut}")
    st.markdown("""
    **Kategori Formzahl (F):**

    - F ≤ 0.25 : Pasang surut harian ganda *(Semidiurnal)*  
    - 0.25 < F ≤ 1.50 : Pasang surut campuran condong ke harian ganda *(Mixed, Predominantly Semidiurnal)*  
    - 1.50 < F ≤ 3.00 : Pasang surut campuran condong ke harian tunggal *(Mixed, Predominantly Diurnal)*  
    - F > 3.00 : Pasang surut harian tunggal *(Diurnal)*
    """)
    st.header("5. Tabel Perhitungan Elevasi Penting")
    st.dataframe(df_elevasi.style.format({
        'Elevasi (m)': lambda x: f"{x:.2f}" if isinstance(x, (int,float)) else x
    }))

    fig_elevasi, ax_elevasi = plt.subplots(figsize=(17, 6))
    ax_elevasi.plot(tanggal, sensor, label='Pasang Surut', color='blue')
    ax_elevasi.axhline(MSL, color='orange', linestyle='-', label='MSL')
    ax_elevasi.axhline(HHWL, color='gray', linestyle='--', label='HHWL')
    ax_elevasi.axhline(LLWL, color='gold', linestyle='--', label='LLWL')
    ax_elevasi.axhline(MHWL, color='green', linestyle='--', label='MHWL')
    ax_elevasi.axhline(MLWL, color='lime', linestyle='--', label='MLWL')
    ax_elevasi.axhline(HWS, color='black', linestyle=':', label='HWS')
    ax_elevasi.axhline(LWS, color='brown', linestyle=':', label='LWS')
    ax_elevasi.axhline(MHWS, color='purple', linestyle='-.', label='MHWS')
    ax_elevasi.axhline(MLWS, color='pink', linestyle='-.', label='MLWS')
    ax_elevasi.set_title('Grafik Pasang Surut + Elevasi Penting')
    ax_elevasi.set_ylabel('Tinggi Permukaan Air (m)')
    ax_elevasi.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax_elevasi.grid(True)
    st.header("Grafik Pasang Surut dengan Elevasi Penting")
    st.pyplot(fig_elevasi)

    st.header("6. Prediksi Pasang Surut")
    timepred_UTIDE = pd.date_range(start=start_pred, end=end_pred, freq=interval)
    tidepred_UTIDE = reconstruct(timepred_UTIDE, decompose_utide, verbose=True)
    h_out_predutide = tidepred_UTIDE.h
    fig_pred, ax_pred = plt.subplots(figsize=(17, 6))
    ax_pred.plot(timepred_UTIDE, h_out_predutide, color='blue', linewidth=1)
    ax_pred.set(xlabel='Tanggal', ylabel='Tinggi Air [m]', title='Prediksi Tinggi Pasang')
    ax_pred.grid()
    st.pyplot(fig_pred)
    df_prediksi = pd.DataFrame({'Tanggal': timepred_UTIDE, 'Elevasi': h_out_predutide})

    tide_utide = reconstruct(tanggal, decompose_utide, verbose=False)
    pred_utide = tide_utide.h
    residual_utide = sensor - pred_utide
    RMSE_UTIDE = math.sqrt(np.mean(np.square(sensor - pred_utide)))
    R_square = r2_score(sensor, pred_utide)

    fig_obs_pred, ax_obs_pred = plt.subplots(figsize=(10, 4))
    ax_obs_pred.plot(tanggal, sensor, label="Observasi", color="black")
    ax_obs_pred.plot(tanggal, pred_utide, label="Prediksi", color="red")
    ax_obs_pred.set_title("Observasi vs Prediksi")
    ax_obs_pred.set_ylabel("Tinggi Air [m]")
    ax_obs_pred.legend()
    ax_obs_pred.grid(True)

    fig_residual, ax_residual = plt.subplots(figsize=(10, 4))
    ax_residual.plot(tanggal, residual_utide, label="Residual", color="green")
    ax_residual.set_title("Residual")
    ax_residual.set_ylabel("Selisih [m]")
    ax_residual.legend()
    ax_residual.grid(True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_obs_pred)
    with col2:
        st.pyplot(fig_residual)

    st.header("7. Ringkasan Analisis")
    st.markdown(f"""
- **Formzahl**: {Formzahl:.4f}  
- **Jenis Pasang Surut**: {jenis_pasang_surut}  
- **RMSE**: {RMSE_UTIDE:.2f}  
- **R²**: {R_square:.2f}  
- **MSL**: {MSL:.2f} m  
- **HWS**: {HWS:.2f} m  
- **LWS**: {LWS:.2f} m  
- **Tunggang Pasang**: {Tunggang_Pasang:.2f} m  
""")

    st.header("8. Ekspor Hasil")

    def fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg')
        buf.seek(0)
        return buf.read()

    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zipf:
            zipf.writestr("Grafik_Observasi.jpg", fig_to_bytes(fig_obs))
            zipf.writestr("Grafik_Formzahl.jpg", fig_to_bytes(fig_formzahl))
            zipf.writestr("Grafik_Observasi_vs_Prediksi.jpg", fig_to_bytes(fig_obs_pred))
            zipf.writestr("Grafik_Residual.jpg", fig_to_bytes(fig_residual))
            zipf.writestr("Grafik_Elevasi_Penting.jpg", fig_to_bytes(fig_elevasi))
            zipf.writestr("Grafik_Prediksi.jpg", fig_to_bytes(fig_pred))
        buffer.seek(0)
        st.download_button("📷 Unduh Semua Grafik JPG (ZIP)", data=buffer, file_name="Grafik_Pasang_Surut.zip", mime="application/zip")

    st.download_button("📄 Unduh Komponen Harmonik", DatFrame_UTide.to_csv(index=False), file_name="Komponen_Harmonik.csv", mime="text/csv")
    st.download_button("📄 Unduh Elevasi Penting", df_elevasi.to_csv(index=False), file_name="Elevasi_Penting.csv", mime="text/csv")
    st.download_button("📅 Unduh Data Prediksi", df_prediksi.to_csv(index=False), file_name="Prediksi_Pasang.csv", mime="text/csv")

# Tanda tangan
st.markdown("""
<h2 style="display:flex;align-items:center;gap:6px; color:#ff9900;">
  <img src="data:image/x-icon;base64,AAABAAEAMDAAAAEAIACoJQAAFgAAACgAAAAwAAAAYAAAAAEAIAAAAAAAgCUAAAAAAAAAAAAAAAAAAAAAAAD+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+//////////////z7+P/26df/8dm3/+3Lmf/qv3v/6Ldh/+azWP/ns1f/6Ldh/+q/e//uypj/8dm4//bp1//8+/j////////////+/v///v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/////v////z9/P/z38X/6L14/+OmGf/jpzH/5a0s/+rBFf/uzwD/8NcA//LbAP/z2wD/8dcA/+7PAP/qwRf/5a4r/+OnMf/jpxj/6L13//Pfxf/8/fz//v////7////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7////8+/n/7dGp/+OmF//hoyX/6bwQ//PfAP/79SD///+L////uP///83//v/b//3+2//9/tv//v/c//7/zP///7f///+M//r1Iv/z3wD/6bwS/+KjJf/jpxf/7dGp//z8+f///////v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/////v////DZu//howD/4KEY/+3OAP/79hn//v+u//394P/89u//+/Hm//rv3//6793/+vDf//vx4f/88eH/+vDf//vv3f/7797/+/Hm//v17//9/eH///+t//r2Gf/uzgD/4aAZ/+GiAP/v2br//v////7////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+///////79/D/5LJg/96YAP/qxAD//Pox//3/yv/79/D/+u/h//vx3v/99+3//vz4/////v////////////////////////////////////7//vz4//z37P/78N7/+u/g//v38P/+/8r//Pox/+rFAP/emAD/5LJf//r38P///////v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7///////Tn1//dmgD/36EC//bqAP///7b/+/fw//ru3P/89ej//v35////////////9/Tx/+fX0P/bwLH/0q6c/86jif/PpIj/1LCZ/9zArv/p183/9/Pu/////////////v36//z15//67tz/+/fx////t//26gD/36EA/92aAP/059b///////7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v//////8N3F/9qPAP/ksQD//P1C//3+6P/67+D//PTk//79+///////+vj2/9m/s/+4dET/rUsA/7FXAP+0XQD/tWAA/7ZgAP+3XwD/uV0A/7lbAP+5WgD/uVQA/8J1AP/cvKX/9/Pv///////9/fr//PPl//rv4f/8/uf//f1B/+OxAP/ajwD/8dzF///////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+///////w28X/14kA/+a8AP///3r/+/nz//vu2v/9+vP///////r49//NqJf/qkIA/69UAP+xWwD/sVoA/7JWAP+zWgD/v3xF/86gg//Yt6P/4Ma2/+XRxP/o1Mf/6NTH/+XQwv/fwKn/1qaA/966mf/28Or///////359P/77tr/+/ny//7/ev/mvAD/2IkA//HbxP///////v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7///////Tl1v/XhQD/5bkA////iv/79vH/+/Dd//79+///////38vD/6hHAP+tVgD/sWEb/7hyQf/Qqpf/49DF//Lr5v/7+PP/+Pb0//Pr5f/u4tr/6drR/+fSxf/lz7//5tDA/+nWyP/u4df/9vDq//7////8/fv///////7////+/fn/+/Hd//v28f///4r/5boA/9aFAP/y5dX///////7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v//////+fbv/9aJAP/fqQD//v96//v28v/78d///v78//7////FmYT/qEUA/7BfIP+yYiL/smIa/7x9Vf+8e0//uWsZ/7RcAP+1WAD/uFoA/7pgAP+7ZAD/vWcA/75qAP+/bAD/wW4A/8NwAP/DcAD/xW8A/8ZwAP/Unmn/6dLA//3+/f///////v78//vx4P/79vH//v97/+CqAP/XiAD/+fXv//7////+/v7//v7+//7+/v/+/v7//v7+//7+/v//////3J9P/9iOAP/8/UL/+/nz//zw3f/+/vz//////8CMcv+lLwD/rFMA/69aAP+xXw3/smIV/7RjDf+2ZhD/t2kZ/7prG/+7bRr/vW8Y/79xFv/Acxb/wXQU/8J1Ev/CdxH/xXoQ/8Z7D//IfQ3/yn4M/8uACP/NfwD/y3oA/82AAP/s18T///////7+/f/78N3//Pnz//39Qv/XjQD/259P///////+/v7//v7+//7+/v/+/v7//v7+///////pz7X/0HUA//TnAP/9/uj/+u7a//78+v/+/v///v7+///////q39r/1rqu/8idif+/hWP/u3xS/7p3Q/+7czb/uGsY/7tsGv+8bhj/vnAW/79yFv/AdBX/wXUT/8J3Ef/EeBD/xnoP/8d7BP/JfAD/yn4A/8uBAP/OggX/z4UI/9CFAP/PewD/37KF///////+/fr/+u7a//3/6P/05wD/0HUA/+nPtf///////v7+//7+/v/+/v7//v////z69//PegD/4bQA////t//67+D//fnz///////////////////////////////////////39fL/6dvU/9Wxnv+7cS3/umsX/7tuGP+9bxf/v3IW/8BzFf/BcwT/wXMA/8FzAP/DcgD/xXYA/8uEIv/Njkj/0JFM/8+ML//NfgD/y3YA/898AP/QgQD/0H4A/9+ue////////fn0//rv4P///7f/4LQA/897AP/8+vf//v////7+/v/+/v7//////+K/oP/KcwD/+/gy//v48f/78+X//////+TVzf+xYCP/uoBj/72DZ/+6eVL/s18A/65OAP+wUQD/s1oA/7VjAP+5axf/vG0Z/75vGP+/cRb/v3IW/8B0Dv/JjVP/0qJ2/9q1l//hxKv/5s++/+jSw//r18f/8OHV//bu5v/9//7///////jz7f/w3c3/472Z/9OCAP/oyKj///////zz5v/79/D/+vkv/8tzAP/hv6D///////7+/v/+/////P37/8hwAP/huwD////K//ru3P/+/Pr//////+rf2v+jKQD/rVcA/69cDv+wXxP/s2Mb/7NmHv+2aB3/t2kd/7hrHP+7bBr/vG4Y/79wF/+/chb/wHQV/8J1E//BdQH/xHUA/8R1AP/FdQD/yHYA/8h3AP/LeAD/y3kA/8x5AP/LdgD/1JI0/+XDpP/28Of////////////x4c7//fz6//79+v/67tz////L/+C7AP/IcAD//P37//7/////////59C+/8NjAP/48g///Pfw//z15///////7+bh///////n2tP/tW9D/6pFAP+tTQD/rlAA/69RAP+yVQD/s1cA/7RYAP+3WwD/uF0A/7pgAP+7YgD/vWYA/79sAP/CdAD/xHYA/8V0AP/EcgD/xnIA/8h0AP/KdQD/y3cA/8x5AP/NewD/z3sA/895AP/OcgD/26Jb//jx5//+/////f7+//7////99ef/+/fw//fzE//CYwD/6NC+////////////zZJe/8+VAP///6//+u/g//78+f/+/vz/oQcA/76Ibv/t4t7///////ry7f/78On/+/Dp//vw6f/78On/+/Hp//zx6f/78ej/+/Ho//zx6f/88ej/9urh/+nSwf/SnnT/0Z1y/+LDrP/y5dz/+PPs//r27//69u7/+vbu//r27v/69u//+/bu//r27v/59e7/4rWG/9eMAP/v2bz///////7+/v/9/Pj/+vDg////rv/PlQD/zpFg///////7+vb/ulUA/+XNAP/+/uD/+vDf////////////qEEA/61YAP+mNwD////+/7Dp//+B3P//iN7//4je//+I3v//iN7//4fe//+H3v//h93//4jb//+J2v//od3//8vs//////////////Dq5v/YwLb/zKaW/8mhkP/KoZD/yqGQ/8qhkP/KopD/zaSP/9Cjif/m1cn/9evd/9yZAP/hogD/8Nm2///////+/v//+/Df//7+4f/lzgD/u1UA//r59v/r3NL/t1IA//bvHf/89u7//ffs//3+/v//////3MS6/6c9AP+uVAD/07Gf//7///8AzP7/AMr//wDL//8Ay///AMv//wDL//8Ay///AMn//wDF//8Awv7/AL/8/wC5/P8swP3/3dfV/6QlAP+rUgD/rFYA/61XAP+tVwD/rFYA/65ZAP+zYAD/uWYA/7xkAP/ev6r/9urb/96bAP/krTP/6LIq//jt3P/+/////Pfs//z27//28Bj/t1IA/+vc0f/cvq7/umAA////jf/78eb///v2/+T2/v+U3/3//////+TTyv+1ZR7/ph0A/+jUyf/p+v//AMb+/wDI//8AyP//AMj//wDI//8Ax///AMT+/wDA/v8Avfz/AL38/wa8/P8Atfv/mdn//797T/+pSwD/qUUA/6hDAP+oQwD/q0kA/7BSAP+7agD/wHQU/8NyAP/hwan//v76//fm0f/y27r/8M6Y/+7Gbv/8/v3//fv2//ry5v///43/uWAA/9y/rv/MoIj/xowA////uP/7797///77/8Xr/v8Axv7/kN79//z/////////8url/93Dtv///fv/9f3+//L+///0/v//9P7///T+///0/v//9P7///T+///2/v//q938/wC4+/8Atvr/Rr///9Sul//eysL/+PX0//v59//7+ff/+/n3//7+/P/Mlmv/xXgA/8l6AP/kxKf/8fr//9zu///7/////////////v/+/Pf//v37//rv3v///7j/xo0A/82fiP/AgmD/0aUA////zf/6793////+/6Ti/v8Ayv//AMf//+b2/f////7///////////////////////Dl4P/q3Nb/6tzW/+rc1v/q3Nb/6tzW/+rc1v/t29P/w97x/wC2+/8DtPr/AK/5/////v/5+Pb/5tnS/+re2P/r3tj/7d/X//Dk2//Ok1z/y38A/8+BAP/ox6f/6/j//1e0+P8Akvb/SK73/6jP+f/k7/z////+//vv3v///83/0aYA/7+CYf+zZDH/17EA///93P/78N///////4nc/v8Ayv//AMn///n+/v///v7//v7+///////fzML/qUkA/6lHAP+pSAD/qUYA/6lGAP+pRgD/qEYA/6lHAP+rNAD/utrx/wC0/P8As/r/ALD4////////////xJF1/7FNAP+6YgD/vmkA/8RwAP/MgQD/0ocD/9WJAP/sz6v////////////t9/z/AJr2/wCY9v+Evfj///////rx3////dz/1rEA/7RkMP+pRwD/2LIA///82//78eL//////3bY/v8Ay///AMn//////v/+/v7//v////Hp5f+lMwD/r14g/61YAP+0bkL/uXlU/7h4U/+5eFP/unpS/719Uf/AeCz/xN3w/wCy+/8Isfr/AKj4//r9/f///////////8WETP+9YgD/xW4A/8p1AP/PewD/1ocA/+OmAP/14L3//v////7+/v////7/AJv2/wCc9f94uPf///////vy4f/++9v/2LIA/6hHAP///////vWt//331//78eH//////3fZ/v8Ay///AMr//////f/+/v7//////9vDuv+rTwD/rloS/72Iav//////////////////////////////////////8/3//wCs+v8Ir/n/AKr5/3nA+P////7//////////v///fj///74///++P///vj////5//79+v/+/v3//f7+//7+/v////7/AJn1/wCb9f95uPf///////vy4f/899f//vWt//7////+/////vO1//331v/78N///////4zd/v8Ay///AMn//+v5/f////7//////+DMw/+qTQD/r18j/61YBv/Inov/zaaU/86nk//QqZL/06uS/9Stkf/XrpD/5LyZ/5rV//8Aqfn/DKv5/wCl+P8Aovb/hL74/5LD+P+Swfj/kb/3/5G99/+Ru/b/lrz2///////+/v///v7+///+/v/4/f7/AJb0/wCZ9P+Ovvf///////rx3//999b//fS1//7////+/////fTJ//30xv/78N3////+/6fj/v8Ayv//AMz//wDJ/v9p1/3/0PD+///7+P+qSQD/rFYA/7BgIf+xXQD/tWEA/7hmAP+9agD/wG8A/8R1AP/IeQD/y3YA/+nHpv9+yv//AKL4/wCk9/8Covf/AJz1/wCY9f8AlvT/AJHz/wCO8/8Ai/L/AIHx/5u99v////7//v7+/////v/p8vz/AJT1/wCW9f+ny/j////+//rw3f/+9MX//fTJ//7////+/v///fjf//3wrv/78N7///77/8jt/v8Ayf//AMz//wDM//8Ay///AMf+/931///78+7/xZN5/69OAP+0VwD/uV0A/7xkAP/AaQD/xG8A/8l0AP/NeQD/0X4A/9N+AP/wxYP/y9/x/2m3+f8Al/b/AJX0/wCR9P8AjfP/AIny/wCF8v8AgfL/AH3x/wBo7//E1vf////+/////v/M4Pr/AJX1/wCT9P/J3fr////8//vw3//8767//fjf//7+///+/v///f35//3nev/78+b///v2/+v5/v8AyP//AMz//yLO/v8Ayf7/AMr+/wDJ/f+36P///f/////////+9/H//vXv//z17v/59O7/+PPu//z17v/+9+////fu//z27//89u////nw///////+//3/8Pb9/+71/P/v9fz/9Pj8//X3/f/09/z/8PX8//T2+//y9vv/+f39/02q9f9vsvf/AJj0/wCO8//q8/3///z2//rz5v/953r//f36//7+///+/v7//v////ziXP/79u3//ffs/////v8bzv7/AMv//zvQ/f/3/f3/sOX9/yvO/f8Axv3/AMT9/1TO/v+E1f7/itX//5fZ/v/e8v7//////7ji/f+Jz/3/fcv9/+j1/v///////v7///7+/v///v7////+/////v/u9/z/g8H5/4vD+f+VyPn/4O/7/5DD+P+Cuvf/N6T0/wCY9f8AmPT/AJf0/yib9P///////fbs//v37f/94Vz//v////7+/v/+/v7//f////3on//89Nz/+/Hf/////v+n4/3/AMr+/wDJ///X8f3////9/6/l/f8Axv3/AMX+/wDB/P8Avv3/AL/8/wC9/P8AuPv/TcP7/we7+v8Atfr/Kbn6/////v///v7//v7+//7+/v/+/v7//v7+/////v+Ixfn/AKH3/wCc9v+52Pr//////////f/V5/v/hLz3/wCa9P8OmfT/AJL0/6nK9///////+/Hf//z03P/86KH//f////7+/v/+/v7//v////324//956H/+/Hg///8+P/x+v7/AMj9/wDL//9g1P3////9/7nm/v8Aw///AMH9/+P1/f+85fz/RMX8/wC4+/8Auvz/ALr7/wC4+/8Atvr/XsD6/9bs+/////3////+/////v/+/v7////+/+Pw+/8Anvf/AKT3/wSl9/////3///7+//7+/v////7////+/06m9P8AlvT/AIjz//H2/f///ff/+/Hg//znov/99uL//v////7+/v/+/v7//v7+//3////82FD/+/fu//z16P//////iNz9/wDK/v8Axv7/0/D8/87t/f8Awv3/AL/9/9/y/P////3////+/+Dy/P+Y1fv/ALj6/wa3+v8Dtfr/ALL6/wCr+v8Esfj/q9b6/+/3/f////7////+/0Wv+P8ApPf/AJv3/8zi+/////7//v7+//7+/v////7/utf4/wCR9P8AkfP/ibr1///////89ej/+/fu//zYUP/9/////v7+//7+/v/+/v7//v7+//3////868H//ezC//rv2//+/fn/8fv9/wDG/v8AyP7/AMf9/97z/P8AwP3/AL79/8Tn/f////7//v7+/////v/z+v3/ALP6/wK1+v8AtPr/AKv5/wCt+f8Arfn/AKn4/wCj+P9yvvj/kMf5/wCi9/8Aovf/crj4/////f/+/v7//v7+/////v/v9fz/AJD0/wmV9P8AifL/8vj9//79+f/68Nv//OzC//zrwf/9/////v7+//7+/v/+/v7//v7+//7+/v/9/////dFD//v57//79OX//////7bm/P8Aw/7/AMT9/2PP/P8Awvz/AL/8/5HV/f////7//v7+/////v+q2vv/ALL6/wCv+v/T6vz/7vb8/6PT+v8Aq/n/AKb4/wCp+P8Apvf/AKP3/w6l9/8AnPb/2er6/////v////7////+//z//f85n/X/AJX0/wCL8v+30vf///////z05P/6+PD//NJE//7////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+////++3N//3kq//68eH//frz/////v9w0vz/AML9/wDB/f8Cwfz/AL/8/wC8+/////7///7+/////f8Atvr/ALP6/wCy+f////3////+/////v////7/1Oj7/2O4+P8Apff/DqX3/wKi9/8Pofb/AJf1/1Ss9v/D2/n/9fv9/1eo9P8Ak/T/AI/z/3Ku9P///////frz//ry4P/85Kr//O7N//7////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v////3Wa//79eX/++/b//79+v/8/v7/Q8f8/wDA/P8FwPz/BL77/wC4/P+94vv////+/9Tr/P8Ar/v/ALD6/6TV+/////7//v7+//3+/v////7/4vD7/wCm9/8Gpff/AJ/2/wqk9v8AmPb/AJv1/wCb9f8AlfX/AJDz/wCW9P8AkfP/RZ7z//z//f///fr/++/a//z15f/81Wv//v////7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7///38+v/80Tf/+/ny//vx3f///vz/+P3+/0PF+/8AvPv/BL38/wC6/P8Atfr/9Pr9/4LL+v8Asfr/AKr5/+v1/P////7////+/////v+83fn/AJ/3/wGl9/8Am/b/pc76/////v/d6/v/j8H3/wCY9f8SmvT/Epj0/wCQ9P9Hn/P/+Pz9///+/P/78d3/+/ny//3QOP/8/fn//v7///7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//3////89eP//NVh//v48v/78t////78//3+/v9zzPv/ALb8/wC5+v8Atvr/H7j6/wC0+v8AsPn/a8H6/////v////7/3O37/2S5+P8Aofj/AKX3/wCb9/+42Pr//////////v/V5vr/aK/3/wCY9P8MmPT/AIz0/3Ww9f/8//3///78//vx3//7+fH//dVi//z14//9/////v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/////O/S//zXcf/7+fL/+/Hd//79+v//////uOH7/wCy+v8As/v/ALT6/way+v8Ar/n/ZL35/4PH+f8Wrvf/AKP4/wCm9/8Epff/J6n3/8ni+v/e7Pv/osv4/w+g9f8AlPX/AJj1/wCU9P8AjfP/utT4///////+/fr/+/Hd//r58f/913b//O/S//3////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v////zv0v/81WD/+/ny//rv2v/9+fP///////P6/f+Pz/r/AKv6/wCt+f8Ar/j/AK35/wCq+P8Aqvj/Dan3/w6m9/8Npff/E6P3/wCd9v8AmPb/AJr0/wCa9f8AlfT/AI3z/4++9v/z+f3///////368//779n/+/ny//3VYv/779H//f////7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7////89eP//NA3//v15f/68eH/+/Tl//79+v//////8/r9/67Y+v84tPr/AKT4/wCl+P8Apff/AKP3/wCi9/8Aofb/AJ72/wCb9v8Al/X/AJL0/zmj9f+u0Pj/9Pj+///////+/fn//PTl//ry4P/79eX//NA2//z14//+/////v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/9/////fz5//zWa//95Kv/+/jw//rw2//89ej///z4/////v//////7/f+/83l/P+v1fr/mcn5/4fB+f+HwPj/mcf5/7DR+f/O4vv/7/f9//////////////z4//z15//78Nz//Pjv//3jq//81mv//fz5//3////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7///3////87c3//dFB//zqwv/7+O7/+/Hg//rx3//89+z///v2/////P////7//////////////////////////v////v///v2//337P/78d7/+vLf//v37//968L//NFC//ztzf/+/////v7///7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/////P////zqwf/90lD//uGi//zy3P/79+3/+vTm//vx3//78d3/+/Hf//vy4f/78uH/+/Hf//rw3f/78d//+/Tm//v37f/88t3//eGi//zST//86cD//P////7////+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//3////+/////fXj//3goP/91Vz//dh7//3lrv/97Mb//PDW//3x1//98db//fDX//7sxv/95a7//dl7//vUXv/94aD//fXi//7////9/////v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v////3////9/////f35//303//97Mn//ue0//7krf/+5K3//ue0//3syf/989///f35//7////9/////v////7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7///7////+/////v////7////+/////v////7////+/////v7///7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v/+/v7//v7+//7+/v8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" 
       alt="Logo Segara GIS" 
       style="width:20px;height:20px;">
  Segara GIS
</h2>
""", unsafe_allow_html=True)















