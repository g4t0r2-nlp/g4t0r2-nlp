# :crocodile: G4T0R2 - Türkçe Doğal Dil İşleme Ekibi

<div align="center">
	<img src="https://avatars.githubusercontent.com/u/171353615?s=200&v=4" alt:"Logo" width="250" height="250" />
</div>

*Bu çalışma, Teknofest 2024 Türkçe Doğal Dil İşleme yarışması "Senaryo" kategorisinde "Entity Bazlı Duygu Analizi" problemi için geliştirilmiştir.*

## :crocodile: Amaç
---

Bu proje, "Entity Bazlı Duygu Analizi Yarışması" kapsamında müşteri geri bildirimlerini analiz ederek belirli hizmet yönleri veya ürün özellikleri ile ilgili duyguları sınıflandırmayı hedeflemektedir. Katılımcılar, farklı sektörlerden ve çeşitli firmalardan veya kurumlardan gelen müşteri yorumlarını inceleyeceklerdir. İlk aşamada, verilen yorumları doğru entity'lere atfetmek için bir yöntem geliştirilmiştir. İkinci aşamada ise bu entity'lere ait hizmet veya ürün özelliklerine yönelik yorumlardaki duyguları (olumlu, olumsuz veya nötr) doğru bir şekilde sınıflandırılmıştır. Bu proje, metin madenciliği ve doğal dil işleme (NLP) tekniklerini kullanarak müşteri geri bildirimlerinden anlamlı içgörüler çıkarma becerilerini test etmeyi amaçlamaktadır. Proje, firmaların müşteri memnuniyetini artırmalarına ve ürün/hizmet geliştirme süreçlerini optimize etmelerine yardımcı olmayı hedeflemektedir.

## :crocodile: Takım Üyeleri
---

**Danışman:** Şengül BAYRAK HAYTA - [LinkedIn](https://www.linkedin.com/in/%C5%9Feng%C3%BCl-bayrak-ba59211b7/) <br/>
**Kaptan:** Alper KARACA - [LinkedIn](https://www.linkedin.com/in/alperkaraca/), [Github](https://github.com/thealper2) <br/>
**Üye:** Ferhat TOSON - [LinkedIn](https://www.linkedin.com/in/ferhattoson/), [Github](https://github.com/ferhattoson) <br/>
**Üye:** Selçuk YAVAŞ - [LinkedIn](https://www.linkedin.com/in/selcukyavas/), [Github](https://github.com/SelcukYavas) <br/>
**Üye:** Mehmet Emin Tayfur - [LinkedIn](https://www.linkedin.com/in/mehmetemintayfur/), [Github](https://github.com/mehmetemintayfur) <br/>

## :crocodile: Gereksinimler
---

Bu proje, Entity Bazlı Duygu Analizi gerçekleştirmek için çeşitli Python kütüphanelerini kullanmaktadır. Aşağıda, kullanılan kütüphaneler ve bunların nasıl kurulacağını gösteren adımlar bulunmaktadır.

```python
app-store-scraper
beautifulsoup4
catboost
fasttext
emoji
gensim
google-play-scraper
lightgbm
matplotlib
mlxtend
numpy
pandas
re
scikit-learn
sklearn-crfsuite
spacy
torch
transformers
tqdm
urllib3
xgboost
```

Çalışmaların hepsi [Kaggle](https://kaggle.com/) ortamında yapılmıştır. Kullanılan donanımlar;

- GPU: 2x8 GB VRAM NVIDIA T4 Ekran kartı
- Processor: Intel(R) Xeon(R) CPU @ 2.20GHz
- RAM: 32 GB

## :crocodile: Veri Seti
---

Yarışma için [Kaggle](https://kaggle.com) ve [HuggingFace](https://huggingface.co/) platformundaki X, Y veri setleri; [SikayetVar](https://www.sikayetvar.com/), [Google Play Store](https://play.google.com/store/games) ve [App Store](https://www.apple.com/tr/app-store/) sitelerinden web kazıma ile veri seti oluşturulmuştur. Veri setine ait dağılımlar aşağıda verilmiştir.
- **RID Sütunu:** Çekilen metine ait eşsiz anahtarı (primary key) belirtir.
- **SID Sütunu:** RID'e ait metindeki kaçıncı cümle olduğunu belirtir.
- **App Sütunu:** Çekilen metinin kaynağını belirtir.
- **Review Sütunu:** Yapılan yorumdaki cümleyi belirtir.
- **Aspect Sütunu:** Cümledeki "Entity" belirtir.
- **Sentiment Sütunu:** Cümledeki "Entity" ait duyguyu belirtir.

| RID | SID | Review | Aspect | Sentiment |
| - | - | - | - | - |
| 1 | 1 |  Uygulamada kartlarda Troy kart geçmiyor. | Troy | Negatif |
| 1 | 2 | Tam 10 yıldır Turkcell kullanıyorum. | Turkcell | Nötr |
| 1 | 3 | Türk Telekom çekim kalitesi çok iyi, tavsiye ederim. | Türk Telekom | Pozitif |

<br/>

 "Entity Extraction" için oluşturulan veri setine ait özellikler aşağıda verilmiştir.

| Sınıf | Veri Sayısı | Açıklama |
| -- | ------- | ----------------- |
| O | 2.085.057 | Entity olmayan kelimeleri temsil eder. |
| B-A | 6.466 | Entity olan kelimeleri temsil eder. |
| I-A | 155.754 | Entity önünde bulunan kelimeleri (sıfatlar, belirteçler vb.) temsil eder. |

<br/>

 "Sentiment Analysis" için oluşturulan veri setine ait özellikler aşağıda verilmiştir.

| Sınıf | Veri Sayısı | Açıklama |
| -- | ------- | ----------------- |
| Negatif | 85.690 | Belirtilen entity'nin duygusunun "Negatif" olduğunu gösterir. |
| Nötr | 8.185 | Belirtilen entity'nin duygusunun "Nötr" olduğunu gösterir. |
| Pozitif | 61.489 | Belirtilen entity'nin duygusunun "Pozitif" olduğunu gösterir. |

## :crocodile: Uygulama
---

### Birinci Uygulama: Girdi olarak verilen metni sınıflandırır.
Bu uygulamada ön yüz geliştirme için JavaScript ile  "React JS"; arka uç geliştirme için "Axios" kullanılmıştır. API için Python ile "FastAPI" kütüphanesi kullanılmıştır. Bu API, Docker konteyneri haline getirilip [HuggingFace](https://huggingface.co/spaces/thealper2/aspect-sentiment-pipeline) platformu üzerinde yayınlanmıştır. <br/>

<div align="center">
	<img src="https://raw.githubusercontent.com/g4t0r2-nlp/g4t0r2-nlp/main/assets/comment-analyser-demo.png?token=GHSAT0AAAAAACTMDUHY2Y2V4PCWCD4XX7JAZVVX55A" />
</div>

### İkinci Uygulama: İnternet üzerinde son girilen yorumları sınıflandırır.

Belirtilen Entity'e ait yorumları, seçilen platform (Google Play Store, App Store, SikayetVar) üzerinden yapılan son yorumları çekerek bu cümlelerdeki Entity'leri bulup duygularını sınıflandırır.

<div align="center">
	<img src="https://raw.githubusercontent.com/g4t0r2-nlp/g4t0r2-nlp/main/assets/gator-search-demo.png?token=GHSAT0AAAAAACTMDUHYL4AEIQ22KOOZSZCMZVVYDOA" />
</div>

## :crocodile: Modeller
---

Her bir problem için eğitilen modeller ve bu modellere ait skorlar aşağıda verilmiştir.

### Aspect Extraction
| Model | Model Tipi | Eğitim Süresi | F1-Macro Skoru |
| ------- | ---------- | -------------- | ------------------ | 
| ConvBERT (dbmdz/convbert-base-turkish-cased) | Cümle Tabanlı | 37 dakika 1 saniye | %99.87 | 
| BERT (dbmdz/bert-base-turkish-cased) | Cümle Tabanlı | 30 dakika 10 saniye | %99.71 |
| BERT (dbmdz/bert-base-turkish-128k-cased) | Cümle Tabanlı | 34 dakika 16 saniye | %99.57 |
| DeBERTa (microsoft/mdeberta-v3-base) | Cümle Tabanlı | 49 dakika 13 saniye | %97.39 |
| ELECTRA (dbmdz/electra-base-turkish-cased-discriminator) | Cümle Tabanlı | 36 dakika 17 saniye | %97.28 |
| DistilBERT (dbmdz/distilbert-base-turkish-cased) | Cümle Tabanlı | 17 dakika 46 saniye | %97.22 |
| CRF | Cümle Tabanlı | 1 dakika 16 saniye | %96.36 |
| ELECTRA (dbmdz/electra-small-turkish-cased-discriminator) | Cümle Tabanlı | 6 dakika 35 saniye | %94.54 |
| Bi-GRU + CRF | Cümle Tabanlı | 9 dakika 55 saniye | %91.21 |
| Bi-LSTM + CRF | Cümle Tabanlı | 10 dakika 1 saniye | %74.75 |
| Bi-LSTM | Cümle Tabanlı | 9 dakika 23 saniye | %72.97 |
| Bi-GRU | Cümle Tabanlı | 9 dakika 33 saniye | %59.00 |

### Sentiment Analysis
| Model | Veri Tabanı | Süre | F1 Macro Skoru |
|-------|-------------|------|----------------|
| BERT (dbmdz/bert-base-turkish-cased) | Kelime Tabanlı | 80 dakika 10 saniye | %84.90 |
| MultinomialNaiveBayes + CountVectorizer | Cümle Tabanlı | 0 dakika 1 saniye | %79.52 |
| MultinomialNaiveBayes + TFIDF | Cümle Tabanlı | 0 dakika 1 saniye | %76.61 |
| BERT (dbmdz/bert-base-turkish-128k-cased) | Kelime Tabanlı | 94 dakika 36 saniye | %76.39 |
| Bi-LSTM (Two Input) | Kelime Tabanlı | 26 dakika 46 saniye | %76.25 |
| LSTM | Cümle Tabanlı | 10 dakika 21 saniye | %75.02 |
| LSTM (Two Input) | Kelime Tabanlı | 23 dakika 2 saniye | %74.56 |
| Attention | Cümle Tabanlı | 35 dakika 5 saniye | %74.84 |
| GRU (Two Input) | Kelime Tabanlı | 24 dakika 12 saniye | %74.84 |
| Bi-GRU (Two Input) | Kelime Tabanlı | 16 dakika 57 saniye | %74.98 |
| Bi-GRU | Cümle Tabanlı | 16 dakika 35 saniye | %74.85 |
| GRU | Cümle Tabanlı | 9 dakika 37 saniye | %74.42 |
| RNN | Cümle Tabanlı | 6 dakika 46 saniye | %74.30 |
| Bi-LSTM | Cümle Tabanlı | 14 dakika 55 saniye | %73.99 |
| FastText | Cümle Tabanlı | 0 dakika 13 saniye | %73.03 |
| RNN (Two Input) | Kelime Tabanlı | 11 dakika 48 saniye | %73.16 |
| LSTM + Word2Vec | Cümle Tabanlı | 9 dakika 13 saniye | %69.69 |
| GRU + Word2Vec | Cümle Tabanlı | 8 dakika 5 saniye | %68.13 |
| Bi-LSTM + Word2Vec | Cümle Tabanlı | 12 dakika 58 saniye | %71.33 |
| LogisticRegression + CountVectorizer | Cümle Tabanlı | 0 dakika 9 saniye | %69.34 |
| Bi-GRU + Word2Vec | Cümle Tabanlı | 11 dakika 13 saniye | %69.65 |
| LSTM + GloVe | Cümle Tabanlı | 8 dakika 47 saniye | %64.41 |
| GRU + GloVe | Cümle Tabanlı | 8 dakika 5 saniye | %64.18 |
| DistilBERT (dbmdz/distilbert-base-turkish-cased) | Kelime Tabanlı | 47 dakika 36 saniye | %64.91 |
| ELECTRA (dbmdz/electra-base-turkish-cased-discriminator) | Kelime Tabanlı | 48 dakika 36 saniye | %65.43 |
| ELECTRA (dbmdz/electra-small-turkish-cased-discriminator) | Kelime Tabanlı | 15 dakika 10 saniye | %56.15 |
| LGBM + CountVectorizer | Cümle Tabanlı | 0 dakika 10 saniye | %55.06 |
| RNN + Word2Vec | Cümle Tabanlı | 2 dakika 29 saniye | %53.42 |
| XGB + CountVectorizer | Cümle Tabanlı | 0 dakika 11 saniye | %53.89 |
| XGB + TFIDF | Cümle Tabanlı | 1 dakika 30 saniye | %52.97 |
| CatBoost + CountVectorizer | Cümle Tabanlı | 3 dakika 27 saniye | %52.08 |
| LogisticRegression + TFIDF | Cümle Tabanlı | 0 dakika 9 saniye | %59.39 |
| LGBM + TFIDF | Cümle Tabanlı | 0 dakika 26 saniye | %53.99 |
| ConvBERT (dbmdz/convbert-base-turkish-cased) | Kelime Tabanlı | 56 dakika 34 saniye | %47.76 |
| RNN + GloVe | Cümle Tabanlı | 1 dakika 14 saniye | %47.44 |
| CatBoost + TFIDF | Cümle Tabanlı | 14 dakika 15 saniye | %51.68 |


## :crocodile: Lisans
---

Uygulamanın lisansına [buradan](https://github.com/g4t0r2-nlp/g4t0r2-nlp/blob/main/LICENSE) ulaşabilirsiniz.

## :crocodile: Kaynaklar
---

- [Neural coreference resolution for Turkish](https://dergipark.org.tr/en/pub/jista/issue/74269/1225097)
- [Marmara Turkish coreference corpus and coreference resolution baseline](https://arxiv.org/abs/1706.01863)
- [An ensemble approach for aspect term extraction in Turkish texts Türkçe metinlerde hedef terimi çıkarımı için bir topluluk yaklaşımı](https://jag.journalagent.com/pajes/pdfs/PAJES-25902-RESEARCH_ARTICLE-SALUR.pdf)
- [Semeval-2016 task 5: Aspect based sentiment analysis](https://biblio.ugent.be/publication/8131987)
- [A hybrid sentiment analysis method for Turkish](https://journals.tubitak.gov.tr/elektrik/vol27/iss3/16/)
- [Application of BiLSTM-CRF model with different embeddings for product name extraction in unstructured Turkish text](https://link.springer.com/article/10.1007/s00521-024-09532-1)
- [Financial named entity recognition for Turkish news texts](https://open.metu.edu.tr/handle/11511/98587)
- [Turkish sentiment analysis using bert](https://ieeexplore.ieee.org/abstract/document/9302492/)
- [So-haTRed: A Novel Hybrid System for Turkish Hate Speech Detection in Social Media With Ensemble Deep Learning Improved by BERT and Clustered-Graph Networks](https://ieeexplore.ieee.org/abstract/document/10559617/)
- [Named entity recognition in Turkish: A comparative study with detailed error analysis](https://www.sciencedirect.com/science/article/pii/S0306457322001674)
