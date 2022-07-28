import nltk
from nltk import word_tokenize, sent_tokenize  
from nltk.tokenize.treebank import TreebankWordDetokenizer
import soundfile as sf
import librosa
import os
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd


nltk.download('words')
nltk.download('punkt')


# NER models
# English NER Model
EN_NER = spacy.load("en_core_web_lg")
EN_NER.add_pipe("merge_entities")

# Arabic NER Model 
tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
AR_NER = pipeline("ner", model=model, tokenizer=tokenizer)


#  Detokenization
detokenizer = TreebankWordDetokenizer()


data = pd.read_csv("./medicine.csv")
custom_labels ={"medicine" :data["medicine"].tolist() , "diseases": data["diseases"].tolist()}


def medical_NER(medical_tokenized_text:list):
    medicalEntities = {}
    for i in range(len(medical_tokenized_text)):
        for label in custom_labels:
            for n in range(len(custom_labels[label])):
                if medical_tokenized_text[i] == custom_labels[label][n]:
                    medicalEntities[medical_tokenized_text[i]] = label
                    medical_tokenized_text[i] = label
    return medicalEntities


def Tokenize(text:str):
    tokenized_text = [list(map(str.lower, word_tokenize(text))) 
        for sent in sent_tokenize(text)] 
    return tokenized_text
      

def Detokenize(tokenized_text:list):
      Anonmized_text= detokenizer.detokenize(tokenized_text)
      Labels = ["GPE", "PER", "LOC", "ORG"]
      encryption = ["Country", "Person", "Location", "Organization"]
      for i in range(len(Labels)):
              Anonmized_text = Anonmized_text.replace(Labels[i] , encryption[i])

      return Anonmized_text



def NER_finder(text:str , mode:str):
  
    tokenized_text = Tokenize(text)
    medicalEntities = {}
    personalEntities = {}

    if mode == "personal_info":
            for i in range(len(tokenized_text[0])):
                # print(tokenized_text[0][i])
                if(tokenized_text[0][i].isascii()):
                    doc = EN_NER(tokenized_text[0][i])
                    if (len(doc.ents) == 1) :
                        if ((doc.ents[0].label_) == "PER" ):
                            # TODO: get personal entities
                            personalEntities[tokenized_text[0][i]] = doc.ents[0].label_

                            tokenized_text[0][i] = doc.ents[0].label_
                else:
                    tag= AR_NER(tokenized_text[0][i])
                    if len(tag) != 0:
                        if((tag[0]['entity'].split("-")[-1]) == "PER"):
                            # TODO: get personal entities
                            personalEntities[tokenized_text[0][i]] = tag[0]['entity'].split("-")[-1]

                            tokenized_text[0][i] = tag[0]['entity'].split("-")[-1]


    # Classify words (AR, EN):
    else:
            medicalEntities = medical_NER(tokenized_text[0])
            for i in range(len(tokenized_text[0])):
                # print(tokenized_text[0][i])

                if(tokenized_text[0][i].isascii()):
                    doc = EN_NER(tokenized_text[0][i])
                    if (len(doc.ents) == 1) :
                        tokenized_text[0][i] = doc.ents[0].label_
                else:
                    tag= AR_NER(tokenized_text[0][i])
                    if len(tag) != 0:
                       tokenized_text[0][i]= tag[0]['entity'].split("-")[-1]

    # elif mode == "medical_stuff":
    #         medical_NER(tokenized_text[0])


    encrypted_text = Detokenize(tokenized_text[0])
    return encrypted_text, medicalEntities, personalEntities
      

# text= "لما hemorrhage في  Egypt احضرت  coffe من عمر"
# print(NER_finder(text, "fully"))
# print(NER_finder(text, "personal_info"))
# print(NER_finder(text, "medical_stuff"))
