#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:42:21 2023

@author: filippooberto
"""

import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.util import minibatch, compounding
import pandas as pd
import random
import ast
import csv

# Caricamento dei dati
data = pd.read_csv('data_entità.csv', delimiter=';')

base_model = spacy.load('it_core_news_sm')

# Inizializzazione dei modelli SpaCy
nlp_org = spacy.blank('it')
ner_org = nlp_org.add_pipe('ner')

nlp_per = spacy.blank('it')
ner_per = nlp_per.add_pipe('ner')

# Preparazione dei dati per l'addestramento separato di ORG e PER
train_data_org = []
train_data_per = []

for index, row in data.iterrows():
    if index < 500:  # Limita a 500 righe per l'addestramento
        text = row['Testo']
        entities = ast.literal_eval(row['Entities']) if pd.notnull(row['Entities']) and row['Entities'] != "[]" else []
        
        entities_for_spacy = [] # Contenitore per le entità trasformate in formato SpaCy
        for entity_text, entity_type in entities:  # Estrai testo e tipo da ogni tupla
            start_pos = text.find(entity_text)  # Trova la posizione di inizio per entità_text nel testo
            if start_pos != -1:  # Se l'entità è trovata nel testo
                end_pos = start_pos + len(entity_text)  # Calcola la posizione di fine
                entities_for_spacy.append((start_pos, end_pos, entity_type))  # Aggiungi alla lista nel formato di SpaCy
                
                # Aggiungi alla lista di train_data per ORG o PER in base al tipo
                if entity_type == 'ORG':
                    train_data_org.append((text, {'entities': [(start_pos, end_pos, 'ORG')]}))
                elif entity_type == 'PER':
                    train_data_per.append((text, {'entities': [(start_pos, end_pos, 'PER')]}))

# Aggiunta delle etichette al modello ORG
for _, annotations in train_data_org:
    for start, end, label in annotations['entities']:
        ner_org.add_label(label)

# Aggiunta delle etichette al modello PER
for _, annotations in train_data_per:
    for start, end, label in annotations['entities']:
        ner_per.add_label(label)

# Funzione di addestramento generica per utilizzo con entrambi i set di dati
def train_ner_model(nlp, train_data):
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # Disabilita tutti i pipe tranne 'ner'
        optimizer = nlp.begin_training()
        for itn in range(100):  # Ad esempio, 100 iterazioni di addestramento
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(f"Losses at iteration {itn}: {losses}")

# Addestramento del modello ORG
train_ner_model(nlp_org, train_data_org)
nlp_org.to_disk('ner_model_org')

# Addestramento del modello PER
train_ner_model(nlp_per, train_data_per)
nlp_per.to_disk('ner_model_per')
            
nlp_model_org = spacy.load('ner_model_org')
nlp_model_per = spacy.load('ner_model_per')

# Funzione per calcolare le metriche di precision, recall e F1 score
def evaluate_model(nlp_model, examples):
    scorer = Scorer()
    for example in examples:  # Ora iteriamo direttamente sugli oggetti Example
        pred_value = nlp_model(example.reference.text)
        example.predicted = pred_value
    scores = scorer.score(examples)  # Passa l'intera lista alla funzione score
    return scores

def remove_duplicate_entities(processed_docs):
    for doc in processed_docs:
        new_ents = []
        seen_ents = set()
        for ent in doc.ents:
            if (ent.start, ent.end, ent.label_) not in seen_ents:
                new_ents.append(ent)
                seen_ents.add((ent.start, ent.end, ent.label_))
        doc.ents = new_ents
    return processed_docs

# Caricamento e preparazione dei dati di validazione
valid_data = pd.read_csv('data_entità.csv', delimiter=';').iloc[500:600]

valid_data_examples_org = []
valid_data_examples_per = []

for _, row in valid_data.iterrows():
    text = row['Testo']
    entities = ast.literal_eval(row['Entities']) if pd.notnull(row['Entities']) and row['Entities'] != "[]" else []
    entities_org = []
    entities_per = []
    for entity_text, entity_type in entities:
        start_pos = text.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            if entity_type == 'ORG':
                entities_org.append((start_pos, end_pos, entity_type))
            elif entity_type == 'PER':
                entities_per.append((start_pos, end_pos, entity_type))

    if entities_org:
        valid_data_examples_org.append((text, {'entities': entities_org}))
    if entities_per:
        valid_data_examples_per.append((text, {'entities': entities_per}))

# Processa i documenti di validazione con i modelli NER e rimuovi entità duplicate
processed_docs_org = [nlp_model_org(text) for text, _ in valid_data_examples_org]
processed_docs_org = remove_duplicate_entities(processed_docs_org)

processed_docs_per = [nlp_model_per(text) for text, _ in valid_data_examples_per]
processed_docs_per = remove_duplicate_entities(processed_docs_per)

# Aggiorna valid_data_examples con le entità pulite
for doc, annot in zip(processed_docs_org, valid_data_examples_org):
    annot[1]['entities'] = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

for doc, annot in zip(processed_docs_per, valid_data_examples_per):
    annot[1]['entities'] = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

# Prepara esempi per la valutazione
valid_examples_org = [Example.from_dict(nlp_org.make_doc(text), annot) for text, annot in valid_data_examples_org]
valid_examples_per = [Example.from_dict(nlp_per.make_doc(text), annot) for text, annot in valid_data_examples_per]

# Valutazione dei modelli
scores_org = evaluate_model(nlp_model_org, valid_examples_org)
scores_per = evaluate_model(nlp_model_per, valid_examples_per)

print("Punteggi per il modello ORG:", scores_org)
print("Punteggi per il modello PER:", scores_per)

# -------------------------- PREDIZIONI -------------------------------

full_data = pd.read_csv('data_entità.csv', delimiter=';')

# Inizializza i dati per le predizioni
predictions_data = []

# Itera sui dati completi
for index, row in full_data.iterrows():
    text = row['Testo']

    # Predici le entità ORG
    org_entities = [(ent.text, ent.label_) for ent in nlp_model_org(text).ents]
    
    # Predici le entità PER
    per_entities = [(ent.text, ent.label_) for ent in nlp_model_per(text).ents]

    # Aggiungi le predizioni al dataset
    predictions_data.append({
        'Nominativo': row['Nominativo'],
        'Testo': text,
        'Entities_ORG': org_entities,
        'Entities_PER': per_entities
    })

# Salva i risultati in un nuovo file CSV
predictions_df = pd.DataFrame(predictions_data)
original_data = pd.read_csv('data_entità.csv', delimiter=';')
merged_df = pd.merge(original_data, predictions_df, how='left', left_index=True, right_index=True, suffixes=('_original', '_predicted'))
merged_df.to_csv('merged_predictions_text_with_names.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)