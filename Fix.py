import streamlit as st
import pandas as pd
from sklearn import preprocessing
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

list_sekolah=['SMAN 1 BABELAN', 'SMAS KARTIKA VIII-1 JAKARTA',
       'SMK NEGERI 2 CILAKU', 'SMAN 1 GEGESIK', 'SMAS FUTURE GATE',
       'SMAN 10 GARUT', 'SMAN 1 CIHAURBEUTI', 'SMKS PLUS AL HASANAH',
       'SMAN 1 KARANGNUNGGAL', 'SMAN 1 ANJATAN',
       'SMKS ISLAMIYAH CIAWI TASIKMALAYA', 'SMA NEGERI 1 PACET',
       'SMKN 1 BANJAR', 'SMAN 3 RANGKAS BITUNG', 'SMAS AL MADINA',
       'SMAS DWI WARNA', 'SMAN 2 CIAMIS', 'SMAN 1 CILIMUS',
       'SMAN 6 GARUT', 'SMAN 10 TANGERANG', 'SMAN 1 PANGANDARAN',
       'SMK BINA LESTARI PUI', 'MAN AWIPARI', 'SMAN 1 TARAJU',
       'SMKN 2 KUNINGAN', 'SMA NEGERI 1 KEDUNGREJA',
       'SMAN 1 SINDANGKERTA', 'SMAN 20 GARUT',
       'SMKS MUHAMMADIYAH TASIKMALAYA', 'SMKN 1 RANCAH',
       'SMAN 3 TASIKMALAYA', 'SMAS YKBBB LELES', 'SMAN 1 CISAYONG',
       'SMAN 6 TASIKMALAYA', 'SMAN 1 CIBUNGBULANG', 'SMAN 5 TASIKMALAYA',
       'SMAN 8 TASIKMALAYA', 'MAS PERSIS 31 BANJARAN',
       'SMAN 1 LEBAKWANGI', 'SMKN 10 GARUT', 'SMAN 1 TASIKMALAYA',
       'SMAN 1 RANCAH', 'SMAS ARIF RAHMAN HAKIM', 'MAN CIANJUR',
       'SMAN 1 SINGAPARNA', 'SMAS TERPADU RIYADLUL ULUM',
       'MAS BUNGURSARI', 'SMKS NURUL WAFA', 'SMAN 1 CIWARINGIN',
       'SMKN 1 TASIKMALAYA', 'SMAN 2 TASIKMALAYA', 'SMAN 9 TASIKMALAYA',
       'SMAN 8 GARUT', 'MAN KIARA KUDA CIAWI', 'SMAS YAPEMRI DEPOK',
       'SMAN 1 MANGUNJAYA', 'MAN CIPASUNG', 'SMAN 15 GARUT',
       'SMKN BANTARKALONG', 'SMKS PELITA CENDEKIA BANGSA',
       'SMAN 4 TASIKMALAYA', 'SMAN 10 TASIKMALAYA',
       'MAS PERSIS 109 KUJANG', 'SMKS AS SAABIQ',
       'SMK MANANGGA PRATAMA TASIKMALAYA', 'SMKN 2 TASIKMALAYA',
       'SMAS MIFTAHUL ULUM', 'MAN TASIKMALAYA',
       'SMAS ASSHIDDIQIYAH KARANGPAWITA', 'SMAN 1 KAWALI',
       'SMAN 1 LAKBOK', 'SMAS PASUNDAN 1 BANDUNG', 'SMAS PLUS NASRUL HAQ',
       'SMAN 2 BANJARSARI', 'SMAN 3 CIAMIS', 'SMAN 1 CISAGA',
       'SMAN 4 CIMAHI', 'MAN RANCAH', 'MAS AL-IHSAN', 'SMAN 1 CIKATOMAS',
       'SMAS HUTAMA', 'SMAS SANDIKTA', 'SMAN 7 TASIKMALAYA',
       'SMAN 1 PARIGI', 'SMAN 1 BANJAR', 'SMAN 92 JAKARTA',
       'SMAN 2 SINGAPARNA', 'SMAS INSAN KAMIL', 'MAN 1 BOGOR',
       'SMAN 4 GARUT', 'SMKS YAPSIPA TASIKMALAYA', 'SMAN 1 GARAWANGI',
       'SMAN 5 JAKARTA', 'SMAN 1 NYALINDUNG', 'SMAS TERPADU DARUSSALAM',
       'SMAN 1 PURWADADI', 'SMKN MANONJAYA']
list_kabupaten=['Kab Bekasi', 'Kota Jakarta Selatan', 'Kab Cianjur', 'Kab Cirebon',
       'Kota Bekasi', 'Kab Garut', 'Kab Ciamis', 'Kab Tasikmalaya',
       'Kab Indramayu', 'Kota Banjar', 'Kab Lebak', 'Kab Bogor',
       'Kab Kuningan', 'Kota Tangerang', 'Kota Tasikmalaya',
       'Kab Cilacap', 'Kab Bandung Barat', 'Kab Bandung',
       'Kota Tangerang Selatan', 'Kota Depok', 'Kota Bandung',
       'Kota Cimahi', 'Kota Jakarta Utara', 'Kota Bogor',
       'Kota Jakarta Pusat', 'Kab Sukabumi', 'Kab Subang']
list_provinsi=['Jawa Barat', 'DKI Jakarta', 'Banten', 'Jawa Tengah']
list_pilihan1=['TEKNIK SIPIL ', 'KESEHATAN MASYARAKAT', 'AGROTEKNOLOGI ',
       'TEKNIK INFORMATIKA', 'PENDIDIKAN BAHASA INGGRIS ', 'MANAJEMEN',
       'AKUNTANSI', 'EKONOMI PEMBANGUNAN',
       'PENDIDIKAN JASMANI, KESEHATAN DAN REKREASI ',
       'PENDIDIKAN BAHASA DAN SASTRA INDONESIA', 'PENDIDIKAN MATEMATIKA',
       'PENDIDIKAN EKONOMI', 'EKONOMI SYARIAH', 'PENDIDIKAN GEOGRAFI',
       'AGRIBISNIS ', 'PENDIDIKAN SEJARAH', 'AGRIBISNIS',
       'PENDIDIKAN BAHASA INGGRIS', 'TEKNIK ELEKTRO', 'GIZI',
       'TEKNIK SIPIL', 'PENDIDIKAN FISIKA', 'AGROTEKNOLOGI',
       'PENDIDIKAN JASMANI, KESEHATAN DAN REKREASI', 'PENDIDIKAN BIOLOGI',
       'ILMU POLITIK', 'PENDIDIKAN LUAR SEKOLAH']
list_pilihan2=['Tidak Memilih', 'AGROTEKNOLOGI ', 'AGRIBISNIS ', 'TEKNIK SIPIL ',
       'EKONOMI SYARIAH', 'MANAJEMEN', 'AKUNTANSI', 'GIZI',
       'TEKNIK ELEKTRO ', 'PENDIDIKAN BAHASA DAN SASTRA INDONESIA',
       'PENDIDIKAN BAHASA INGGRIS ', 'PENDIDIKAN MATEMATIKA',
       'EKONOMI PEMBANGUNAN', 'PENDIDIKAN GEOGRAFI', 'PENDIDIKAN SEJARAH',
       'KESEHATAN MASYARAKAT', 'TEKNIK SIPIL', 'ILMU POLITIK',
       'TEKNIK INFORMATIKA', 'PENDIDIKAN EKONOMI', 'AGRIBISNIS',
       'PENDIDIKAN FISIKA', 'PENDIDIKAN BIOLOGI',
       'PENDIDIKAN BAHASA INGGRIS', 'AGROTEKNOLOGI',
       'PENDIDIKAN LUAR SEKOLAH',
       'PENDIDIKAN JASMANI, KESEHATAN DAN REKREASI']

list_ranking=[178,  29, 7, 195, 20, 148, 138,  42, 256, 151,  30,  73, 157,
       128,  41,  59, 252, 307, 149, 213,  22, 131, 124, 112, 106,  66,
        23, 192, 167, 259,  92,  44, 182, 170, 224, 183, 103, 102, 101,
        95, 130,  58, 143, 125,  35,  17,  72, 109, 158, 168,  96,  80,
       335, 160, 217,  71,  27, 161, 177,  48, 126,  12, 121, 196, 188,
       204, 219, 184,  76,  40, 180,  50, 187, 132, 155, 185, 220, 137,
        61, 153, 104, 169,  62, 191,  45, 162, 144,  83,  36,  97]


# Judul aplikasi Streamlit
st.title("Aplikasi Klasifikasi Lulus Pendaftaran Perguruan Tinggi")

# Tab bar untuk Klasifikasi
tab = st.sidebar.selectbox("Pilihan Tab", ["Data Training", "Klasifikasi", "Evaluasi"])

# Load X_train data
data_training = pd.read_excel('X_train.xlsx')
data = pd.read_excel('Mahasiswa.xlsx')

if tab == "Data Training":
    st.subheader("Data Training:")
    st.write(data_training)
    st.write("Jumlah Data: ", len(data_training))
    # Tampilkan jumlah fitur (kolom)
    num_features = data.shape[1]
    st.write(f"Jumlah Fitur (Kolom): {num_features}")

elif tab == "Klasifikasi":
    st.subheader("Klasifikasi:")

    # Fungsi encoding
    def encoding(data):
        df = pd.concat([data_training, data], ignore_index=True)
        label_encoder = preprocessing.LabelEncoder()

        columns_to_encode = ['JK', 'bidikmisi', 'Sekolah', 'Kabupaten', 'Provinsi', 'Pilihan 1', 'Pilihan 2']

        for column in columns_to_encode:
            df[column] = label_encoder.fit_transform(df[column])

        last = df.iloc[[-1]]
        return last

    # Fungsi xgb
    def xgb(data):
        loaded_model = joblib.load('XGboost.pkl')
        pred = loaded_model.predict(data)
        return pred[0]

    # Input nilai dari pengguna
    col1, col2 = st.columns(2)

    # Kolom 1
    jk = col1.selectbox("Jenis Kelamin", ["P", "L"])
    bidikmisi = col1.selectbox("Bidikmisi", ["Bidik Misi", "Reguler"])
    sekolah = col1.selectbox("Sekolah:", list_sekolah)
    kabupaten = col1.selectbox("Kabupaten:", list_kabupaten)
    provinsi = col1.selectbox("Provinsi:", list_provinsi)
    pilihan_1 = col1.selectbox("Pilihan 1:", list_pilihan1)
    pilihan_2 = col1.selectbox("Pilihan 2:", list_pilihan2)
    ranking = col1.selectbox("Ranking Sekolah:", list_ranking)
    nilai_mapel_un = col1.number_input("Nilai Mapel UN:")

    # Kolom 2
    x1 = col2.number_input("X1:(Rentang 9.7-12.1)", min_value=9.7, max_value=12.1)
    x2 = col2.number_input("X2:(Rentang 9.6-12.1)", min_value=9.6, max_value=12.1)
    x3 = col2.selectbox("X3:", [1, 1.1, 1.025, 1.05])
    x5 = col2.number_input("X5:(Rentang 9.6-12.1)", min_value=9.6, max_value=12.1)
    x6 = col2.number_input("X6:(Rentang 9.6-12.1)", min_value=9.6, max_value=12.1)
    x7 = col2.number_input("X7:(Rentang 9.6-12.1)", min_value=9.6, max_value=12.1)
    x8 = col2.number_input("X8:(Rentang 9.6-12.1)", min_value=9.6, max_value=12.1)
    x9 = col2.selectbox("X9:", [7.5, 9.375, 10., 10.625, 11.25, 6.25])
    xt = col2.number_input("XT:(Rentang 66-84)", min_value=66, max_value=84)
    # Button untuk melakukan prediksi
    classify_button = st.button("Klasifikasi")

    if classify_button:
        input_data = pd.DataFrame({
            "JK": [jk],
            "bidikmisi": [bidikmisi],
            "Sekolah": [sekolah],
            "Kabupaten": [kabupaten],
            "Provinsi": [provinsi],
            "Pilihan 1": [pilihan_1],
            "Pilihan 2": [pilihan_2],
            "Ranking Sekolah": [ranking],
            "Nilai Mapel UN": [nilai_mapel_un],
            "X1": [x1],
            "X2": [x2],
            "X3": [x3],
            "X5": [x5],
            "X6": [x6],
            "X7": [x7],
            "X8": [x8],
            "X9": [x9],
            "XT": [xt],
        })

        # Encoding data
        encoded_data = encoding(input_data)

        # Prediksi menggunakan model XGBoost
        prediksi = xgb(encoded_data)

        st.subheader("Hasil Prediksi:")
        if prediksi == 0:
            st.write("Mohon maaf anda tidak diterima")
        elif prediksi == 1:
            st.write("Selamat, Anda diterima di pilihan 1")
        elif prediksi == 2:
            st.write("Selamat, Anda diterima di pilihan 2")

elif tab == "Evaluasi":
        XGB = xgb.XGBClassifier( objective='multi:softmax',num_class=3 )
        X_train = data.drop(["Lulus Pilihan"], axis=1)
        Y_train = data[["Lulus Pilihan"]]
        trained=XGB.fit(X_train,Y_train)
        k_values = [2,3,4,6,8]
        # Fungsi evaluasi dengan K-fold
        def evaluate_with_kfold(X_train, Y_train, k_values, metrics):
            result = []

            for metric in metrics:
                st.write(f"\nEvaluating model based on {metric}-score:")

                for k in k_values:
                    kf = KFold(n_splits=k, shuffle=True, random_state=42)

                    XGB = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

                    # Define the scoring metric
                    if metric == 'accuracy':
                        scoring_metric = 'accuracy'
                    elif metric == 'f1':
                        scoring_metric = make_scorer(f1_score, average='macro')
                    elif metric == 'precision':
                        scoring_metric = make_scorer(precision_score, average='macro')
                    elif metric == 'recall':
                        scoring_metric = make_scorer(recall_score, average='macro')

                    scores = cross_val_score(XGB, X_train, Y_train, cv=kf, scoring=scoring_metric)

                    avg_score = np.mean(scores)
                    result.append({'K': k, 'Metric': metric, 'Average Score': avg_score})

            return pd.DataFrame(result)

        # Tab evaluasi
        def evaluation_tab():
            st.subheader("Evaluasi:")

            # Dropdown untuk memilih metrik evaluasi
            metrics = ['accuracy', 'f1', 'precision', 'recall']  # Sesuaikan dengan kebutuhan Anda
            selected_metric = st.selectbox("Pilih Metrik Evaluasi:", metrics)

            # Menambahkan tombol untuk melakukan evaluasi
            evaluate_button = st.button("Evaluasi Model")

            if evaluate_button:
                # Load the XGB model
                loaded_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
                loaded_model.fit(X_train, Y_train)

                # Evaluasi model dengan K-fold
                eval_results_df = evaluate_with_kfold(X_train, Y_train, k_values, [selected_metric])

                # Menampilkan hasil evaluasi
                st.subheader("Hasil Evaluasi:")
                st.table(eval_results_df)

        # Panggil fungsi evaluasi pada tab
        evaluation_tab()