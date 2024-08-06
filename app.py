from fastapi.middleware.cors import CORSMiddleware
import json
import torch
import nltk
nltk.download("punkt")
from transformers import BertTokenizerFast, BertTokenizer, BertForTokenClassification, BertForSequenceClassification, Pipeline
from nltk import sent_tokenize
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Uygulama HuggingFace uzerinde canliya alinmistir. Asagidaki linklerden istek atilabilir.
# HuggingFace Repo: https://huggingface.co/spaces/thealper2/aspect-sentiment-pipeline
# HuggingFace API Linki: https://thealper2-aspect-sentiment-pipeline.hf.space/predict/
# Docs icin : https://thealper2-aspect-sentiment-pipeline.hf.space/docs

# Ornek kullanim:
# curl -d '{"text": "Turkcell cok iyidir."}' -H "Content-Type: application/json" -X POST https://thealper2-aspect-sentiment-pipeline.hf.space/predict/
# Sonuc:s
# {"entity_list":["Turkcell"],"results":[{"aspect":"Turkcell","sentiment":"olumlu"}]}

class AspectSentimentPipeline(Pipeline):
    def __init__(self, aspect_extraction_model, aspect_extraction_tokenizer, aspect_sentiment_model, aspect_sentiment_tokenizer, device):
        super().__init__(aspect_extraction_model, aspect_extraction_tokenizer)
        self.aspect_extraction_model = aspect_extraction_model
        self.aspect_extraction_tokenizer = aspect_extraction_tokenizer
        self.aspect_sentiment_model = aspect_sentiment_model
        self.aspect_sentiment_tokenizer = aspect_sentiment_tokenizer
        self.device = device

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        return sent_tokenize(inputs)

    def _forward(self, sentences):
        main_results = []
        main_aspects = []
        for sentence in sentences:
            aspects = self.extract_aspects(sentence, self.aspect_extraction_model, self.aspect_extraction_tokenizer, self.device)
            for aspect in aspects:
                main_aspects.append(aspect)
                sentiment = self.predict_sentiment(sentence, aspect)
                main_results.append({"aspect": aspect, "sentiment": sentiment})
                
        return {"entity_list": main_aspects, "results": main_results}

    def postprocess(self, model_outputs):
        return model_outputs

    def predict_sentiment(self, sentence, aspect):
        inputs = self.aspect_sentiment_tokenizer(aspect, sentence, return_tensors="pt").to(self.device)
        self.aspect_sentiment_model.to(self.device)
        self.aspect_sentiment_model.eval()

        with torch.no_grad():
            outputs = self.aspect_sentiment_model(**inputs)
            logits = outputs.logits

        sentiment = torch.argmax(logits, dim=-1).item()
        sentiment_label = self.aspect_sentiment_model.config.id2label[sentiment]
        sentiment_id_to_label = {
            "LABEL_0": "olumsuz",
            "LABEL_1": "nötr",
            "LABEL_2": "olumlu"
        }

        return sentiment_id_to_label[sentiment_label]

    def align_word_predictions(self, tokens, predictions):
        aligned_tokens = []
        aligned_predictions = []
        for token, prediction in zip(tokens, predictions):
            if not token.startswith("##"):
                aligned_tokens.append(token)
                aligned_predictions.append(prediction)
            else:
                aligned_tokens[-1] = aligned_tokens[-1] + token[2:]
        return aligned_tokens, aligned_predictions

    def extract_aspects(self, review, aspect_extraction_model, aspect_extraction_tokenizer, device):
        inputs = self.aspect_extraction_tokenizer(review, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=64, return_tensors="pt").to(device)
        self.aspect_extraction_model.to(device)
        self.aspect_extraction_model.eval()
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = self.aspect_extraction_model(ids, attention_mask=mask)
            logits = outputs[0]
        
        active_logits = logits.view(-1, self.aspect_extraction_model.num_labels) 
        flattened_predictions = torch.argmax(active_logits, axis=1) 
        
        tokens = self.aspect_extraction_tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        ids_to_labels = {0: 'O', 1: 'B-A', 2: 'I-A'}
        token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    
        filtered_tokens = [token for token in tokens if token not in ["[PAD]", "[CLS]", "[SEP]"]]
        filtered_predictions = [pred for token, pred in zip(tokens, token_predictions) if token not in ["[PAD]", "[CLS]", "[SEP]"]]
        
        aligned_tokens, aligned_predictions = self.align_word_predictions(filtered_tokens, filtered_predictions)
    
        aspects = []
        current_aspect = []
            
        for token, prediction in zip(aligned_tokens, aligned_predictions):
            if prediction == "B-A":
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []
                current_aspect.append(token)
            elif prediction == "I-A":
                if current_aspect:
                    current_aspect.append(token)
            else:
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []
        
        if current_aspect:
            aspects.append(" ".join(current_aspect))
    
        return aspects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aspect_extraction_model = BertForTokenClassification.from_pretrained("thealper2/aspect-extraction-model")
aspect_extraction_tokenizer = BertTokenizerFast.from_pretrained("thealper2/aspect-extraction-tokenizer")

aspect_sentiment_model = BertForSequenceClassification.from_pretrained("thealper2/aspect-sentiment-model")
aspect_sentiment_tokenizer = BertTokenizer.from_pretrained("thealper2/aspect-sentiment-tokenizer")

pipeline = AspectSentimentPipeline(
    aspect_extraction_model=aspect_extraction_model,
    aspect_extraction_tokenizer=aspect_extraction_tokenizer,
    aspect_sentiment_model=aspect_sentiment_model,
    aspect_sentiment_tokenizer=aspect_sentiment_tokenizer,
    device=device
)

app = FastAPI()

class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz.  Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.get("/", tags=["Home"])
def api_home():
    return {"detail": "Welcome to FastAPI!"}

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    result = pipeline(item.text)
    return result


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)