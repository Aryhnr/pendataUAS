import streamlit as st
import pandas as pd
from pyngrok import ngrok
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Menu Data
def menu_data():
    st.subheader('Data')
    st.write('Di sini Anda dapat memanipulasi data. Anda dapat melakukan operasi seperti membaca data dari file, menampilkan ringkasan statistik, melakukan visualisasi data, dan lain sebagainya.')
    st.write('Data diambil dari website kaagle berikut link dari website tersebut : https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset')
    st.write('Ada beberapa tipe data diantaranya ada type data kategory, bolean, dan numerik')
    st.write('Data berisi nilai laboratorium donor darah dan pasien Hepatitis C dan nilai demografis seperti usia.')
    df = pd.read_csv('HepatitisCdata.csv')
    df
# Menu Preprocessing Data
def menu_preprocessing_data():
    st.subheader('Preprocessing Data')
    preprocessing_option = st.radio('Pilihan Preprocessing', ['Min-Max Scaler','Simple Imputer'])
    if preprocessing_option == 'Min-Max Scaler':
        st.write('Anda memilih Min-Max Scaler')

        # Memanggil data hepatitis C
        data = pd.read_csv('HepatitisCdata.csv')

        # Memilih kolom numerik untuk dilakukan scaling
        numeric_columns = ['Age', 'ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        data_numeric = data[numeric_columns]

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_numeric)

        # Membuat DataFrame baru untuk data yang telah di-scaling
        data_scaled = pd.DataFrame(data_scaled, columns=data_numeric.columns)

        # Menampilkan data setelah scaling
        st.subheader('Data Setelah Scaling')
        st.dataframe(data_scaled)
    elif preprocessing_option == 'Simple Imputer':
        st.write('Anda memilih Simple Imputer')

        # Memanggil data hepatitis C
        data = pd.read_csv('HepatitisCdata.csv')

        # Memilih kolom numerik untuk dilakukan scaling
        numeric_columns = ['Age', 'ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        data_numeric = data[numeric_columns]

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_numeric)
        imputer = SimpleImputer(strategy='most_frequent')  # Ganti strategi imputasi sesuai kebutuhan
        data_X = imputer.fit_transform(data_numeric)
        # Membuat DataFrame baru untuk data yang telah di-scaling
        data_X = pd.DataFrame(data_X, columns=data_numeric.columns)

        # Menampilkan data setelah scaling
        st.subheader('Data Setelah Simple Impunter')
        st.dataframe(data_X)

# Menu Modelling
def menu_modelling():
    st.subheader('Modelling')
    model_options = ['Naive Bayes', 'Decision Tree', 'MLP', 'KNN']
    selected_model = st.radio('Pilih Algoritma', model_options)
    st.subheader('Akurasi')
    if selected_model == 'Naive Bayes':
        data = pd.read_csv('HepatitisCdata.csv')
        data_kategorikal = data[['Sex']]
        data_numerik = data[['Age', 'ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']] 
        data_kategorikal_encoded = pd.get_dummies(data_kategorikal)
        data_preprocessed = pd.concat([data_kategorikal_encoded, data_numerik], axis=1)
        data_X = data_preprocessed
        data_y = data['Category']
        imputer = SimpleImputer(strategy='most_frequent')  # Ganti strategi imputasi sesuai kebutuhan
        data_X = imputer.fit_transform(data_X)
        scaler = MinMaxScaler()
        data_X = scaler.fit_transform(data_X)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracy
    elif selected_model == 'Decision Tree':
        st.write('Anda memilih Decision Tree')
        data = pd.read_csv('HepatitisCdata.csv')
        data_kategorikal = data[['Sex']]
        data_numerik = data[['Age', 'ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']] 
        data_kategorikal_encoded = pd.get_dummies(data_kategorikal)
        data_preprocessed = pd.concat([data_kategorikal_encoded, data_numerik], axis=1)
        data_X = data_preprocessed
        data_y = data['Category']
        imputer = SimpleImputer(strategy='most_frequent')  # Ganti strategi imputasi sesuai kebutuhan
        data_X = imputer.fit_transform(data_X)
        scaler = MinMaxScaler()
        data_X = scaler.fit_transform(data_X)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        # Prediksi label untuk data uji
        y_pred = model.predict(X_test)

        # Evaluasi skor akurasi
        accuracy = accuracy_score(y_test, y_pred)
        accuracy
    
    elif selected_model == 'MLP':
        st.write('Anda memilih MLP')
        data = pd.read_csv('HepatitisCdata.csv')
        data_kategorikal = data[['Sex']]
        data_numerik = data[['Age', 'ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']] 
        data_kategorikal_encoded = pd.get_dummies(data_kategorikal)
        data_preprocessed = pd.concat([data_kategorikal_encoded, data_numerik], axis=1)
        data_X = data_preprocessed
        data_y = data['Category']
        imputer = SimpleImputer(strategy='most_frequent')  # Ganti strategi imputasi sesuai kebutuhan
        data_X = imputer.fit_transform(data_X)
        scaler = MinMaxScaler()
        data_X = scaler.fit_transform(data_X)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy
    elif selected_model == 'KNN':
        st.write('Anda memilih KNN')
        data = pd.read_csv('HepatitisCdata.csv')
        data_kategorikal = data[['Sex']]
        data_numerik = data[['Age', 'ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']] 
        data_kategorikal_encoded = pd.get_dummies(data_kategorikal)
        data_preprocessed = pd.concat([data_kategorikal_encoded, data_numerik], axis=1)
        data_X = data_preprocessed
        data_y = data['Category']
        imputer = SimpleImputer(strategy='most_frequent')  # Ganti strategi imputasi sesuai kebutuhan
        data_X = imputer.fit_transform(data_X)
        scaler = MinMaxScaler()
        data_X = scaler.fit_transform(data_X)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy
    
# Menu Implementasi
def menu_implementasi():
    st.subheader('Implementasi')
    data = pd.read_csv('HepatitisCdata.csv')
    algorithm = st.selectbox("Select Algorithm", ["Naive Bayes", "Decision Tree", "MLP", "KNN"])
# Memilih fitur kategorikal dan numerik
    data_kategorikal = data[['Sex']]
    data_numerik = data[['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]

    # Melakukan one-hot encoding pada data kategorikal
    data_kategorikal_encoded = pd.get_dummies(data_kategorikal)

    # Menggabungkan data kategorikal dan numerik
    data_preprocessed = pd.concat([data_kategorikal_encoded, data_numerik], axis=1)

    # Memisahkan fitur dan target
    data_X = data_preprocessed
    data_y = data['Category']

    # Melakukan preprocessing menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    data_X_scaled = scaler.fit_transform(data_X)

    # Melakukan imputasi pada data yang masih memiliki missing value
    imputer = SimpleImputer(strategy='most_frequent')
    data_X_imputed = imputer.fit_transform(data_X_scaled)
    st.write('Masukkan data untuk melakukan prediksi')

    # Input fitur kategorikal
    input_sex = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    if input_sex == 'Male':
        input_data = [1, 0]
    else:
        input_data = [0, 1]

    # Input fitur numerik
    input_data.append(st.number_input('Age'))
    input_data.append(st.number_input('ALB'))
    input_data.append(st.number_input('ALP'))
    input_data.append(st.number_input('ALT'))
    input_data.append(st.number_input('AST'))
    input_data.append(st.number_input('BIL'))
    input_data.append(st.number_input('CHE'))
    input_data.append(st.number_input('CHOL'))
    input_data.append(st.number_input('CREA'))
    input_data.append(st.number_input('GGT'))
    input_data.append(st.number_input('PROT'))
    input_data_scaled = scaler.transform([input_data])
    st.subheader('Akurasi')
    if algorithm == "Naive Bayes":
        model = GaussianNB()
        model.fit(data_X_imputed, data_y)
        predicted_class = model.predict(input_data_scaled)
        predicted_class[0]
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(data_X_imputed, data_y)
        predicted_class = model.predict(input_data_scaled)
        predicted_class[0]
    elif algorithm == "MLP":
        model = MLPClassifier()
        model.fit(data_X_imputed, data_y)
        predicted_class = model.predict(input_data_scaled)
        predicted_class[0]
    elif algorithm == "KNN":
        model = KNeighborsClassifier()
        model.fit(data_X_imputed, data_y)
        predicted_class = model.predict(input_data_scaled)
        predicted_class[0]


# Judul halaman
st.title('Aplikasi Data Science')

with st.sidebar:
    selected_menu = option_menu("Main Menu", ['Data','Preprocessing Data','Modelling','Implementasi'], 
         default_index=0)


# Tampilkan konten sesuai menu yang dipilih
if selected_menu == 'Data':
    menu_data()
elif selected_menu == 'Preprocessing Data':
    menu_preprocessing_data()
elif selected_menu == 'Modelling':
    menu_modelling()
elif selected_menu == 'Implementasi':
    menu_implementasi()