import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

### 
def visualisasiFitur(x, y, name, test=None):
    x_condition1 = []
    x_condition2 = []
    y_condition1 = []
    y_condition2 = []
    for i, kelas in enumerate(y):
        if kelas == 0:
            x_condition1.append(x[i])
            y_condition1.append(y[i])
        else:
            x_condition2.append(x[i])
            y_condition2.append(y[i])
            
    # Membuat gambar 
    plt.figure("Persebaran fitur {}".format(name))
    plt.plot(
        x_condition1,
        y_condition1,
        marker="o",
        color="blue",
        linestyle="",
        label="Tidak Diabetes",
    )
    plt.plot(
        x_condition2,
        y_condition2,
        marker="o",
        linestyle="",
        color="red",
        label="Diabetes",
    )

    plt.xlabel(name)
    plt.ylabel("Kelas")
    plt.ylim(-1, 2)
    plt.yticks([-1, 0, 1, 2], ["", "Tidak Diabetes", "Diabetes", ""])

    plt.legend()
    
    # Simpan hasil gambar
    plt.savefig("grafik_fitur/{}".format(name))

    
### Program jalan pertama dari sini bukan dari atas
### Program jalan pertama dari sini bukan dari atas
### Program jalan pertama dari sini bukan dari atas

# Membaca file Excel
data = pd.read_csv('diabetes.csv')

# Menampilkan data diabetes
print("===============================================================================")
print("Data Diabetes")
print("===============================================================================")
print(data)
print("")

# Memisahkan fitur (X) dan target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Membagi data menjadi data latih dan data uji dengan pembagian data latih 80% dan data uji 20%
# X_train => fitur data latih
# X_test => fitur data test
# y_train => kelas data latih
# y_test => kelas data latih

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Visualisasi persebaran fitur pada data latih


# membuat list fitur dalam variabel columns berdasarkan title dataset
columns = X_train.columns.tolist()


for col in columns:
    visualisasiFitur(X_train[col].tolist(), y_train.tolist(), col)

# Buat objek KNN
knn = KNeighborsClassifier(n_neighbors=5, p=2)

# Latih model dengan data latih
knn.fit(X_train, y_train)

# Lakukan prediksi terhadap data uji
y_pred = knn.predict(X_test)
print(y_pred)
# Membuat tabel confusion matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(
    cm,
    index=[i for i in ["tidak diabetes", "diabetes"]],
    columns=[i for i in ["pred tidak diabetes", "pred diabetes"]],
)
plt.figure("Confusion Matrix")
sns.set(font_scale=1.3)
sns.heatmap(df_cm, annot=True)
plt.savefig("confusion tabel/confusion_matrix.png")

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)


# Print hasil akurasi
print("Akurasi prediksi: {:.2f}%".format(accuracy * 100))

# Menampilkan semua gambar
# plt.show()
