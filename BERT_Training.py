#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:06:47 2023

@author: filippooberto
"""

import spacy
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification

class NominativoDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#Setting dati di partenza
df = pd.read_excel("File Crediti Filippo - Copy.xlsx")
df.drop_duplicates(subset=['N. O.'], inplace=True)

relevant_columns = [
    'Nominativo', 'Utilizzo', 'Garanzie', 'Soci', 'Attività', 'Scopo', 'Ragione sociale' 
]

corrected_relevant_df = df[relevant_columns].copy()

vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = vectorizer.fit_transform(corrected_relevant_df["Nominativo"].fillna(''))

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_similarities = np.clip(cosine_similarities, 0, 1)

clustering = DBSCAN(eps=0.5, min_samples=2, metric="precomputed").fit(1 - cosine_similarities)

corrected_relevant_df['cluster_label'] = clustering.labels_

correction_map = corrected_relevant_df.groupby('cluster_label')['Nominativo'].apply(
    lambda x: x.value_counts().idxmax() if x.nunique() > 1 else x.iloc[0]
).to_dict()

if -1 in correction_map:
    del correction_map[-1]
    
corrected_relevant_df["Nominativo"] = corrected_relevant_df["Nominativo"].replace(correction_map)

corrected_relevant_df.to_csv('Data_Cleaned.csv', index=False, encoding='utf-8-sig')


# Preparazione dei dati

data = pd.read_csv('Data_Cleaned.csv', delimiter=',')
data = data.dropna(subset=['Nominativo'])

data['Ragione sociale'] = data['Ragione sociale'].apply(lambda x: 'Persona' if 'persona fisica' in str(x).lower() else 'Azienda')

# Mappatura delle etichette: 'Persona' -> 0, 'Azienda' -> 1
label_mapping = {'Persona': 0, 'Azienda': 1}
data['labels'] = data['Ragione sociale'].map(label_mapping)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 100

# Conversione delle etichette in numeri
label_mapping = {'Persona': 0, 'Azienda': 1}
train_data['labels'] = train_data['Ragione sociale'].map(label_mapping)
val_data['labels'] = val_data['Ragione sociale'].map(label_mapping)
test_data['labels'] = test_data['Ragione sociale'].map(label_mapping)

# Creazione dei Dataset
train_dataset = NominativoDataset(train_data['Nominativo'].to_numpy(), train_data['labels'].to_numpy(), tokenizer, max_len)
val_dataset = NominativoDataset(val_data['Nominativo'].to_numpy(), val_data['labels'].to_numpy(), tokenizer, max_len)
test_dataset = NominativoDataset(test_data['Nominativo'].to_numpy(), test_data['labels'].to_numpy(), tokenizer, max_len)

# Creazione dei DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Caricamento del modello BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # 2 etichette: Persona, Azienda

# Addestramento del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Funzione per l'addestramento
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Addestrare per un numero di epoche
num_epochs = 4
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device)
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
def eval_model(model, data_loader, device):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            losses.append(loss.item())

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

val_acc, val_loss = eval_model(model, val_data_loader, device)
print(f'Validation loss {val_loss} accuracy {val_acc}')

test_acc, test_loss = eval_model(model, test_data_loader, device)
print(f'Test loss {test_loss} accuracy {test_acc}')

torch.save(model.state_dict(), 'bert_model.pth')

#-------------------- APPLICAZIONE MODELLO --------------------

def predict(model, text, tokenizer, max_len):
    model.eval()  # Assicurati che il modello sia in modalità di valutazione
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


def predict_and_extract_with_name(model, text, tokenizer, max_len, nominativo):
    prediction = predict(model, text, tokenizer, max_len)
    return {'Nominativo': nominativo, 'Testo': text, 'Tipo': 'Persona Fisica' if prediction == 0 else 'Azienda'}

# Lista per raccogliere i risultati
risultati = []

for index, row in data.iterrows():
    for column in ['Utilizzo', 'Garanzie', 'Soci', 'Attività', 'Scopo']:
        text = row[column]
        if pd.notna(text):  # Controlla se il testo non è NaN
            result = predict_and_extract_with_name(model, text, tokenizer, max_len, row['Nominativo'])
            risultati.append(result)

# Creazione del DataFrame
df_risultati = pd.DataFrame(risultati)

# Salvataggio nel file CSV
df_risultati.to_csv('risultati_classificazione.csv', index=False)

 #-------------------- UTILIZZO NER PER ESTRAZIONE ENTITÀ --------------------

nlp = spacy.load('xx_ent_wiki_sm')

data = pd.read_csv('risultati_classificazione.csv')

# Funzione per estrarre le entità da una colonna di un dataframe
def extract_entities(df, text_column):
    entities = []
    for doc in nlp.pipe(df[text_column]):
        ents = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in ['PER', 'ORG']]
        entities.append(ents)
    return entities

# Applicare la funzione al dataframe
data['Entities'] = extract_entities(data, 'Testo')

# Visualizzare i risultati
print(data[['Testo', 'Entities']])

data.to_csv('data_entità.csv', index=True)
