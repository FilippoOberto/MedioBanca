#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:42:37 2023

@author: filippooberto
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
import pandas as pd
import random
import ast

def load_data(filepath, start_row, end_row):
    data = pd.read_csv(filepath, delimiter=';').iloc[start_row:end_row]
    return data

def prepare_data(data, entity_type, from_nominativo=False):
    prepared_data = []
    for _, row in data.iterrows():
            text = row['Testo_original']
            entities = ast.literal_eval(row[f'Entities_{entity_type}']) if pd.notnull(row[f'Entities_{entity_type}']) else []
            for entity_text, _ in entities:
                start_pos = text.find(entity_text)
                if start_pos != -1:
                    end_pos = start_pos + len(entity_text)
                    prepared_data.append((text, {'entities': [(start_pos, end_pos, entity_type)]}))
    return prepared_data

def continue_training_ner(nlp, train_data):
    optimizer = nlp.resume_training()
    optimizer.learn_rate = 1e-7
    for itn in range(1000):
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=compounding(4.0, 16.0, 1.005)):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.6, sgd=optimizer, losses=losses)
        print(f"Losses at iteration {itn}: {losses}")

def evaluate_model(nlp_model, validation_data):
    scorer = Scorer()
    examples = []
    for text, annotations in validation_data:
        doc_gold_text = nlp_model.make_doc(text)
        example = Example.from_dict(doc_gold_text, annotations)
        pred_value = nlp_model(example.reference.text)
        example.predicted = pred_value
        examples.append(example)
    scores = scorer.score(examples)
    return scores

#Main script
# Caricamento dei modelli pre-addestrati
nlp_org = spacy.load('ner_model_org')
nlp_per = spacy.load('ner_model_per')

# Caricamento dei dati di addestramento e validazione (se necessario)
training_data = load_data('merged_predictions_text_with_names.csv', 0, 850)
validation_data = load_data('merged_predictions_text_with_names.csv', 850, 900)

# Preparazione dei dati di addestramento (se necessario)
train_data_org = prepare_data(training_data, 'ORG')
train_data_per = prepare_data(training_data, 'PER')

# Preparazione dei dati di validazione (se necessario)
valid_data_org = prepare_data(validation_data, 'ORG')
valid_data_per = prepare_data(validation_data, 'PER')

# Continua l'addestramento dei modelli
continue_training_ner(nlp_org, train_data_org)
continue_training_ner(nlp_per, train_data_per)

# Valutazione dei modelli (se necessario)
model_org_scores = evaluate_model(nlp_org, valid_data_org)
model_per_scores = evaluate_model(nlp_per, valid_data_per)

print("Punteggi modello ORG:", model_org_scores)
print("Punteggi modello PER:", model_per_scores)

# Salva i modelli addestrati (se necessario)
nlp_org.to_disk('updated_org_model')
nlp_per.to_disk('updated_per_model')

def analyze_errors(nlp_model, validation_data):
    errors = []

    for text, annotations in validation_data:
        doc = nlp_model.make_doc(text)
        example = Example.from_dict(doc, annotations)
        true_entities = example.reference.ents
        predicted_entities = example.predicted.ents

        for true_ent in true_entities:
            if true_ent not in predicted_entities:
                errors.append((text, true_ent, 'Missed'))

        for pred_ent in predicted_entities:
            if pred_ent not in true_entities:
                errors.append((text, pred_ent, 'False Positive'))

    return errors

# Utilizza la funzione per analizzare gli errori per ciascun modello
errors_org = analyze_errors(nlp_org, valid_data_org)
errors_per = analyze_errors(nlp_per, valid_data_per)

print(errors_org)
print(errors_per)