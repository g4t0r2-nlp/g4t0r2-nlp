# :crocodile: G4T0R2 - Türkçe Doğal Dil İşleme Ekibi

<div align="center">
	<img src="https://avatars.githubusercontent.com/u/171353615?s=200&v=4" alt:"Logo" width="250" height="250" />
</div>

*Bu çalışma, Teknofest 2024 Türkçe Doğal Dil İşleme yarışması "Senaryo" kategorisinde "Entity Bazlı Duygu Analizi" problemi için geliştirilmiştir.*

### :crocodile: Amaç
---

Bu proje, "Entity Bazlı Duygu Analizi Yarışması" kapsamında müşteri geri bildirimlerini analiz ederek belirli hizmet yönleri veya ürün özellikleri ile ilgili duyguları sınıflandırmayı hedeflemektedir. Katılımcılar, farklı sektörlerden ve çeşitli firmalardan veya kurumlardan gelen müşteri yorumlarını inceleyeceklerdir. İlk aşamada, verilen yorumları doğru entity'lere atfetmek için bir yöntem geliştirilmiştir. İkinci aşamada ise bu entity'lere ait hizmet veya ürün özelliklerine yönelik yorumlardaki duyguları (olumlu, olumsuz veya nötr) doğru bir şekilde sınıflandırılmıştır. Bu proje, metin madenciliği ve doğal dil işleme (NLP) tekniklerini kullanarak müşteri geri bildirimlerinden anlamlı içgörüler çıkarma becerilerini test etmeyi amaçlamaktadır. Proje, firmaların müşteri memnuniyetini artırmalarına ve ürün/hizmet geliştirme süreçlerini optimize etmelerine yardımcı olmayı hedeflemektedir.

### :crocodile: Takım Üyeleri
---

**Danışman:** Şengül BAYRAK HAYTA - [LinkedIn]()
**Kaptan:** Alper KARACA - [Github]()
**Üye:** Ferhat TOSON - [Github]()
**Üye:** Selçuk YAVAŞ - [Github]()
**Üye:** Mehmet Emin Tayfur - [Github]()

### :crocodile: Gereksinimler
---

Bu proje, Entity Bazlı Duygu Analizi gerçekleştirmek için çeşitli Python kütüphanelerini kullanmaktadır. Aşağıda, kullanılan kütüphaneler ve bunların nasıl kurulacağını gösteren adımlar bulunmaktadır.

```python
pandas==2.0
matplotlib==3.7.1
nltk==3.8.1
wordcloud==1.8.2.2
scikit-learn==1.2.2
torch==2.0.0
transformers==4.27.4
numpy==1.24.2
tqdm==4.65.0
more-itertools==9.1.0
```

Çalışmaların hepsi [Kaggle](https://kaggle.com/) ortamında yapılmıştır. Kullanılan donanımlar;

- GPU: 2x8 GB VRAM'li T4 Ekran kartı
- Processor: CUDA 11.2
- RAM: Intel Xeon işlemci

### :crocodile: Veri Seti
---

Yarışma için [Kaggle]() ve [HuggingFace]() platformundaki X, Y veri setleri; [SikayetVar](), [Google Play Store]() ve [App Store]() sitelerinden web kazıma ile veri seti oluşturulmuştur. Veri setine ait dağılımlar aşağıda verilmiştir.
- **RID Sütunu:** Çekilen metine ait eşsiz anahtarı (primary key) belirtir.
- **SID Sütunu:** RID'e ait metindeki kaçıncı cümle olduğunu belirtir.
- **App Sütunu:** Çekilen metinin kaynağını belirtir.
- **Review Sütunu:** Yapılan yorumdaki cümleyi belirtir.
- **Aspect Sütunu:** Cümledeki "Entity" belirtir.
- **Sentiment Sütunu:** Cümledeki "Entity" ait duyguyu belirtir.

| RID | SID | Review | Aspect | Sentiment |
| - | - | - | - | - |
| 1 | 1 | Metin | Turkcell | Pozitif |
| 1 | 2 | Metin | Turk Telekom | Nötr |
| 1 | 3 | Metin | Vodafone | Negatif |

- "Entity Extraction" için oluşturulan veri setine ait özellikler aşağıda verilmiştir.

| Sınıf | Veri Sayısı | Açıklama |
| -- | ------- | ----------------- |
| O | 123.123 | Açıklama buradadır. |
| B-A | 123.123 | Açıklama buradadır. |
| I-A | 123.123 | Açıklama buradadır. |

- "Sentiment Analysis" için oluşturulan veri setine ait özellikler aşağıda verilmiştir.

| Sınıf | Veri Sayısı | Açıklama |
| -- | ------- | ----------------- |
| Negatif | 123.123 | Açıklama buradadır. |
| Nötr | 123.123 | Açıklama buradadır. |
| Pozitif | 123.123 | Açıklama buradadır. |

### :crocodile: Uygulama
---

##### Birinci Uygulama: Girdi olarak verilen metni sınıflandırır.
Bu uygulamada ön yüz geliştirme için JavaScript ile  "React JS"; arka uç geliştirme için "Axios" kullanılmıştır. API için Python ile "FastAPI" kütüphanesi kullanılmıştır. Bu API, Docker konteyneri haline getirilip [HuggingFace]() platformu üzerinde yayınlanmıştır. 

##### İkinci Uygulama: İnternet üzerinde son girilen yorumları sınıflandırır.



### :crocodile: Modeller
---

Her bir problem için eğitilen modeller ve bu modellere ait skorlar aşağıda verilmiştir.

#### Aspect Extraction
| Model | Model Tipi | Eğitim Süresi | F1-Macro Skoru |
| ------- | ---------- | -------------- | ------------------ | 
| BERT (dbmdz/bert-base-turkish-128k-cased) | Cümle Tabanlı | 34 dakika 16 saniye | %99.57 |
| BERT (dbmdz/bert-base-turkish-cased) | Cümle Tabanlı | 30 dakika 10 saniye | %99.71 |
| Bi-GRU + CRF | Cümle Tabanlı | 9 dakika 55 saniye | %91.21 |
| Bi-GRU | Cümle Tabanlı | 9 dakika 33 saniye | %59.00 |
| Bi-LSTM + CRF | Cümle Tabanlı | 10 dakika 1 saniye | %74.75 |
| Bi-LSTM | Cümle Tabanlı | 9 dakika 23 saniye | %72.97 |
| CRF | Cümle Tabanlı | 1 dakika 16 saniye | %96.36 |
| ConvBERT (dbmdz/convbert-base-turkish-cased) | Cümle Tabanlı | 37 dakika 1 saniye | %99.87 | 
| DeBERTa (microsoft/mdeberta-v3-base) | Cümle Tabanlı | 49 dakika 13 saniye | %97.39 |
| DistilBERT (dbmdz/distilbert-base-turkish-cased) | Cümle Tabanlı | 17 dakika 46 saniye | %97.22 |
| ELECTRA (dbmdz/electra-base-turkish-cased-discriminator) | Cümle Tabanlı | 36 dakika 17 saniye | %97.28 |
| ELECTRA (dbmdz/electra-small-turkish-cased-discriminator) | Cümle Tabanlı | 6 dakika 35 saniye | %94.54 |

#### Sentiment Analysis
| Model | Model Tipi | Eğitim Süresi | F1-Macro Skoru |
| -------- | -------------- | ----------- | ------------------ | 
| Attention | Cümle Tabanlı | 35 dakika 5 saniye | %74.84 |
| BERT (dbmdz/bert-base-turkish-128k-cased) | Kelime Tabanlı | 94 dakika 36 saniye | %76.39 |
| BERT (dbmdz/bert-base-turkish-cased) | Kelime Tabanlı | 80 dakika 10 saniye | %86.52 |
| Bi-GRU (Two Input) | Kelime Tabanlı | 16 dakika 57 saniye | %74.98 |
| Bi-GRU | Cümle Tabanlı | 16 dakika 35 saniye | %74.85 |
| Bi-GRU + GloVe | Cümle Tabanlı | 11 dakika 4 saniye | %67.39 |
| Bi-GRU + Word2Vec | Cümle Tabanlı | 11 dakika 13 saniye | %69.65 | 
| Bi-LSTM (Two Input) | Kelime Tabanlı | 26 dakika 46 saniye | %76.25 | 
| Bi-LSTM | Cümle Tabanlı | 14 dakika 55 saniye | %73.99 |
| Bi-LSTM + GloVe | Cümle Tabanlı | 12 dakika 46 saniye | %67.80 |
| Bi-LSTM + Word2Vec | Cümle Tabanlı | 12 dakika 58 saniye | %71.33 |
| CatBoost + CountVectorizer | Cümle Tabanlı | 3 dakika 27 saniye | %52.08 |
| CatBoost + TFIDF | Cümle Tabanlı | 14 dakika 15 saniye | %51.68 |
| ConvBERT (dbmdz/convbert-base-turkish-cased) | Kelime Tabanlı | 56 dakika 34 saniye | %47.76 |
| DistilBERT (dbmdz/distilbert-base-turkish-cased) | Kelime Tabanlı | 47 dakika 36 saniye | %64.91 |
| ELECTRA (dbmdz/electra-base-turkish-cased-discriminator) | Kelime Tabanlı | 48 dakika 36 saniye | %65.43 | 
| ELECTRA (dbmdz/electra-small-turkish-cased-discriminator) | Kelime Tabanlı | 15 dakika 10 saniye | %56.15 |
| FastText | Cümle Tabanlı | 0 dakika 13 saniye | %73.03 |
| GRU (Two Input) | Kelime Tabanlı | 24 dakika 12 saniye | %74.84 |
| GRU | Cümle Tabanlı | 9 dakika 37 saniye | %74.42 |
| GRU + GloVe | Cümle Tabanlı | 8 dakika 5 saniye | %64.18 | 
| GRU + Word2Vec | Cümle Tabanlı | 8 dakika 5 saniye | %68.13 |
| LGBM + CountVectorizer | Cümle Tabanlı | 0 dakika 10 saniye | %55.06 |
| LGBM + TFIDF | Cümle Tabanlı | 0 dakika 26 saniye | %53.99 |
| LSTM (Two Input) | Kelime Tabanlı | 23 dakika 2 saniye | %74.56 |
| LSTM | Cümle Tabanlı | 10 dakika 21 saniye | %75.02 |
| LSTM + GloVe | Cümle Tabanlı | 8 dakika 47 saniye | %64.41 |
| LSTM + Word2Vec | Cümle Tabanlı | 9 dakika 13 saniye | %69.69 |
| LogisticRegression + CountVectorizer | Cümle Tabanlı | 0 dakika 9 saniye | %69.34 |
| LogisticRegression + TFIDF | Cümle Tabanlı | 0 dakika 9 saniye | %59.39 |
| MultinomialNaiveBayes + CountVectorizer | Cümle Tabanlı | 0 dakika 1 saniye | %79.52 |
| MultinomialNaiveBayes + TFIDF | Cümle Tabanlı | 0 dakika 1 saniye | %76.61 |
| RNN (Two Input) | Kelime Tabanlı | 11 dakika 48 saniye | %73.16 |
| RNN | Cümle Tabanlı | 6 dakika 46 saniye | %74.30 |
| RNN + GloVe | Cümle Tabanlı | 1 dakika 14 saniye | %47.44 |
| RNN + Word2Vec | Cümle Tabanlı | 2 dakika 29 saniye | %53.42 |
| XGB + CountVectorizer | Cümle Tabanlı | 0 dakika 11 saniye | %53.89 |
| XGB + TFIDF | Cümle Tabanlı | 1 dakika 30 saniye | %52.97 |

### :crocodile: Kaynaklar
- [Kaynak 1]()
- [Kaynak 2]()
- [Kaynak 3]()
- [Kaynak 4]()

