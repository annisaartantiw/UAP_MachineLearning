
# **Deep Learning untuk Klasifikasi Gambar Buah dan Sayur**

## **✨ Overview Project**
Proyek ini bertujuan untuk mengembangkan sebuah sistem klasifikasi gambar yang dapat mengenali dan membedakan berbagai jenis buah dan sayur. Sistem ini dapat digunakan dalam berbagai aplikasi yang membutuhkan kemampuan untuk mengenali jenis buah dan sayur dari gambar, seperti aplikasi belanja atau sistem manajemen inventaris makanan.

- **Link Dataset yang digunakan:** [Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- **Repositori GitHub:** [UAP_MachineLearning](https://github.com/annisaartantiw/UAP_MachineLearning)

## **✨ Preprocessing dan Modelling**

### **Preprocessing**
Preprocessing yang dilakukan antara lain adalah resizing gambar ke ukuran (150, 150), normalisasi (1./255), dan augmentasi data untuk meningkatkan keragaman dataset dengan metode rotasi, pergeseran, pemotongan, zoom, dan flipping horizontal. Setelah preprocessing, dataset dibagi menjadi 3 bagian: Training Set, Validation Set, dan Test Set.

### **Model yang digunakan**
Proyek ini menggunakan dua model deep learning untuk klasifikasi gambar:
1. **Model CNN Sederhana**
2. **Model VGG16 Pretrained**

## **✨ CNN Architecture**
Model CNN yang digunakan terdiri dari beberapa lapisan:
- 3 Lapisan Konvolusi (Conv2D) dengan ukuran filter bertambah (32, 64, 128)
- Lapisan Pooling (MaxPooling2D) untuk mengurangi dimensi
- Lapisan Flatten untuk meratakan data dan dilanjutkan dengan lapisan Dense untuk klasifikasi

### **Model CNN Sederhana**
Model CNN dibangun dengan 3 lapisan konvolusi dan menggunakan dropout untuk mengurangi overfitting. Model ini disiapkan untuk melatih data dan melakukan evaluasi terhadap akurasi.

```python
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])
```

## **✨ VGG16 Architecture**
Model VGG16 merupakan model pretrained yang digunakan dengan memanfaatkan bobot yang sudah dilatih sebelumnya pada dataset ImageNet. VGG16 akan digunakan sebagai dasar untuk model klasifikasi buah dan sayur.

```python
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
```

### **Modelling dan Evaluasi**
- **CNN Model:** Melatih model CNN dengan 10 epoch, menggunakan data yang sudah di-augmentasi.
- **VGG16 Model:** Melatih model VGG16 dengan 10 epoch menggunakan transfer learning, di mana hanya lapisan baru yang dilatih.

## **✨ Hasil Evaluasi**
### **Plot Akurasi dan Loss**

#### **Model CNN**
Plot berikut menunjukkan akurasi pelatihan dan validasi selama proses training untuk model CNN.

![Model CNN Training Accuracy](cnn_accuracy_plot.png)

#### **Model VGG16**
Plot berikut menunjukkan akurasi pelatihan dan validasi selama proses training untuk model VGG16.

![Model VGG16 Training Accuracy](vgg_accuracy_plot.png)

### **Evaluasi Akurasi Model pada Data Pengujian**
Setelah pelatihan, kedua model diuji pada data uji, dan hasil evaluasi menunjukkan akurasi sebagai berikut:
- **CNN Test Accuracy:** 85%
- **VGG16 Test Accuracy:** 90%

### **Classification Report**
Laporan klasifikasi untuk kedua model setelah diuji pada dataset test:

#### **CNN Model**
```plaintext
Classification Report for CNN:
              precision    recall  f1-score   support

        Apple       0.88      0.85      0.87       500
        Banana      0.87      0.89      0.88       500
        Orange      0.86      0.84      0.85       500
        ...
```

#### **VGG16 Model**
```plaintext
Classification Report for VGG16:
              precision    recall  f1-score   support

        Apple       0.91      0.92      0.91       500
        Banana      0.90      0.92      0.91       500
        Orange      0.91      0.88      0.89       500
        ...
```

## **✨ Kesimpulan**
Model VGG16 yang menggunakan transfer learning dari ImageNet menghasilkan akurasi yang lebih tinggi dibandingkan model CNN sederhana, terutama dalam hal kemampuan generalisasi pada data uji. Namun, model CNN sederhana tetap memberikan hasil yang baik dan dapat menjadi pilihan ketika membutuhkan model yang lebih ringan.

## **✨ Link Ke Depan**
Ke depan, proyek ini dapat dikembangkan lebih lanjut dengan meningkatkan kualitas data, menggunakan model yang lebih canggih seperti ResNet atau DenseNet, dan menerapkan fine-tuning pada model pretrained untuk hasil yang lebih optimal.

- **Link Dataset:** [Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- **Repositori GitHub:** [UAP_MachineLearning](https://github.com/annisaartantiw/UAP_MachineLearning)
