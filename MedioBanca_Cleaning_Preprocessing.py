import pandas as pd
import numpy as np
import Levenshtein

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import DBSCAN
from fuzzywuzzy import process
from scipy.spatial.distance import pdist, squareform

#DEFINIZIONE DI FUNZIONI
#Funzione che converte le date in oggetti 'datetime'
def convert_float_to_datetime(date_float):
	if pd.isna(date_float): #Controllare i valori NaN
		return None
	try:
		date_str = str(int(date_float)) #Convertire i float in int e poi in str
		return datetime.strptime(date_str, '%Y%m%d')
	except ValueError:
		return None

def clean_text(text):
	if pd.isna(text) or not isinstance(text, str): #Controllare i valori NaN
		return text
	return text.lower().strip()

def format_numbers(number):
    if pd.isna(number) or not isinstance(number, (int, float)):
        return number
    return '{:,.0f}'.format(number).replace(',', ' ')

def correct_name_frequency(serie):
    counts = serie.value_counts() #Ottiene mappa di frequenza per i valori nella serie
    rare_name = counts[counts < 5].index #Filtra i nomi che appaiono meno di un certo threshold
    for rare_name in rare_name:
        similar_name, score = process.extractOne(rare_name, counts.index)
        if score > 85:
            serie.replace(rare_name, similar_name, inplace=True)
    return serie

def compute_distance_matrix(strings): 
    strings = list(strings)
    
    # Converte l'array di stringhe in una matrice 2D
    strings_matrix = np.array(strings)[:, np.newaxis]
    
    # Calcola le distanze utilizzando pdist e la distanza di Levenshtein
    distances = pdist(strings_matrix, lambda u, v: Levenshtein.distance(u[0], v[0]))
    
    # Converti il vettore di distanze in una matrice e restituiscila
    return squareform(distances)


#Carico il DataFrame
df = pd.read_excel("File Crediti Filippo - Copy.xlsx")

#Rimuovo i duplicati basati sul numero d'ordine
df.drop_duplicates(subset=['N. O.'], inplace=True)

#Correggo e ottimizzo la colonna "esercizio"
df["anno_fiscale"] = "19" + df.esercizio.astype(str).str[:2] + "-19" + df.esercizio.astype(str).str[2:]
df.drop(['esercizio'], axis=1)

#Unisco e pulisco le colonne dell'importo
df["importo"] = df['Dec.'].astype(str) + df["Importo"].astype(str)
df["importo"] = df['importo'].str.replace('nan', '').str.strip()
df.drop(['Dec.', 'Importo'], axis=1)

#Converto l'importo in numero e identifico una colonna "note_rifiuto"
df['valore_importo'] = pd.to_numeric(df.importo, errors='coerce')
df['rifiuto'] = ~df['importo'].str.isdigit()

#Seleziono le colonne rilevanti
corrected_relevant_columns = [
    'anno_fiscale', 'Fonte', 'N. O.', 'N. P.', 'Data Arr.', 'Nominativo', 'Piazza', 'Provincia',
    'C.d.A.', 'Utilizzo', 'Garanzie', 'Rimborso da', 'Scadenza', 'Contratto', 'Ragione sociale',
    'Capitale', 'Valuta', 'Soci', 'Attività', 'Condizioni C/C int.', 'Condizioni C/C TUS min.',
    'Commissione C/C', 'Condizioni sc. Pagherò int.', 'Condizioni sc. Pagherò TUS min',
    'Commissioni sc. Pagherò', 'Commissione assicurato finanziamento', 'Scopo', 'Consistenza',
    'Fatturato anno 1', 'Fatturato anno 2', 'Fatturato anno 3', 'Anno 1', 'Anno 2', 'Anno 3', 
    'Utili anno 1', 'Utili anno 2', 'Utili anno 3', 'Export in', 'Filiale', 'valore_importo', 
    'rifiuto', 'importo'
]

corrected_relevant_df = df[corrected_relevant_columns].copy()

#Converto date in formato datetime
date_columns = ['Data Arr.', 'C.d.A.', 'Rimborso da', 'Scadenza', 'Contratto', 'Anno 1', 'Anno 2', 'Anno 3']

for column in date_columns:
    corrected_relevant_df[column] = pd.to_datetime(corrected_relevant_df[column], format='%Y%m%d', errors='coerce')        

#Pulizia del testo
text_columns = [
    'Fonte', 'Nominativo', 'Piazza', 'Provincia', 'Utilizzo', 'Garanzie', 'Ragione sociale',
    'Valuta', 'Soci', 'Attività', 'Export in', 'Filiale', 'rifiuto'
]

for column in text_columns:
	corrected_relevant_df[column] = corrected_relevant_df[column].apply(clean_text)


#Formattazione numeri
numeric_columns = [
    'N. O.', 'N. P.', 'Capitale', 'Consistenza', 
    'Fatturato anno 1', 'Fatturato anno 2', 'Fatturato anno 3',
    'Utili anno 1', 'Utili anno 2', 'Utili anno 3', 'valore_importo',
    'importo'
]

scaler = MinMaxScaler()

# Prima, converti i valori in numeri e scala
for col in numeric_columns:
    # Converti in numeri
    corrected_relevant_df[col] = pd.to_numeric(
        corrected_relevant_df[col].astype(str).str.replace(r'\D', '', regex=True), 
        errors='coerce'
    )
    
    # Scala i valori numerici
    if corrected_relevant_df[col].notna().any():  # Aggiunto un check per assicurarsi che ci siano valori non-NaN
        corrected_relevant_df[col + '_scaled'] = scaler.fit_transform(
            corrected_relevant_df[[col]].fillna(0)  # Puoi scegliere un modo migliore per gestire i NaN se necessario
        )

# Poi, formatta i numeri (considera che questo converte i numeri in stringhe)

for col in numeric_columns:
    corrected_relevant_df[col] = corrected_relevant_df[col].apply(format_numbers)


#Clustering per disambiguare 'Nominativo'
"""
noms = df["Nominativo"].unique()
distance_matrix = compute_distance_matrix(noms)
"""
#Uso TF-IDF per convertire i nomi in vettori
vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = vectorizer.fit_transform(df["Nominativo"].fillna(''))

#Calcolo la similiratità del coseno
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

#Mi assicuro che i valori siano compresi tra 0 e 1
cosine_similarities = np.clip(cosine_similarities, 0, 1)

#Uso DBSCAN per il clustering basato sulla similarità del coseno
clustering = DBSCAN(eps=0.5, min_samples=2, metric="precomputed").fit(1 - cosine_similarities)

corrected_relevant_df['cluster_label'] = clustering.labels_

correction_map = corrected_relevant_df.groupby('cluster_label')['Nominativo'].apply(lambda x: x.value_counts().idxmax() if x.nunique() > 1 else x.iloc[0]).to_dict()

if -1 in correction_map:
    del correction_map[-1]
            
corrected_relevant_df["Nominativo"] = corrected_relevant_df["Nominativo"].replace(correction_map)

#Gestisco i valori nulli
threshold = 0.75
df = df.loc[:, df.isnull().mean() < threshold]

#Esporta il DataFrame pulito
corrected_relevant_df.to_csv('Data_Cleaned.csv', index=False, encoding='utf-8-sig')