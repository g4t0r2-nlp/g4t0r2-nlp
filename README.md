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

**Danışman:** Şengül BAYRAK HAYTA - [LinkedIn]() <br/>
**Kaptan:** Alper KARACA - [Github]() <br/>
**Üye:** Ferhat TOSON - [Github]() <br/>
**Üye:** Selçuk YAVAŞ - [Github]() <br/>
**Üye:** Mehmet Emin Tayfur - [Github]() <br/>

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

- GPU: 2x8 GB VRAM'li T4 Ekran kartı <br/>
- Processor: CUDA 11.2 <br/>
- RAM: Intel Xeon işlemci <br/>

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
| Model | Eğitim Süresi | F1-Macro Skoru |
| ------- | -------------- | ------------------ | 
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |

#### Sentiment Analysis
| Model | Eğitim Süresi | F1-Macro Skoru |
| ------- | -------------- | ------------------ | 
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |
| Birinci model | 10 dakika | 99.99 |

### :crocodile: Kaynaklar
- [Kaynak 1]()
- [Kaynak 2]()
- [Kaynak 3]()
- [Kaynak 4]()

