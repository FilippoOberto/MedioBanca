#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:09:13 2023

@author: filippooberto
"""

import folium
import time
import base64
import branca
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from folium.plugins import MarkerCluster
from PIL import Image

#------------------ FUNZIONI ------------------

#Ottiene le coordinate per la colonna 'Piazza'

location_cache = {}

def get_location_by_name(piazza, provincia=None):
    geolocator = Nominatim(user_agent="MedioBanca_Progetto", timeout=10)
    query = piazza if provincia is None else f"{piazza}, {provincia}"
    
    # Controlla se la query è già stata effettuata
    if query in location_cache:
        return location_cache[query]

    for _ in range(3):  # Prova 3 volte
        try:
            location = geolocator.geocode(query)
            if location:
                # Memorizza il risultato nella cache
                location_cache[query] = (location.latitude, location.longitude)
                return location.latitude, location.longitude
            elif provincia is not None:
                location = geolocator.geocode(piazza)
                if location:
                    return location.latitude, location.longitude
        except (GeocoderUnavailable, GeocoderTimedOut):
            time.sleep(2)  # Aspetta 2 secondi prima di ritentare in caso di errore
        time.sleep(1)
    return None

export_location_cache = {}

def get_export_location(row):
    # Geolocalizza utilizzando sia la capitale che il codice ISO del paese
    if pd.notnull(row['Capitale(2)']) and pd.notnull(row['Export in']):
        location = get_location_by_name(f"{row['Capitale(2)']}, {row['Export in']}")
        return location
    return None

#Offset per i marker di operazioni ed esportazioni

existing_operation_coordinates = set()
existing_export_coordinates = set()

def add_offset_if_needed(lat, lon, is_export=False):
    lat_offset = 0.050
    lon_offset = 0.050
    existing_coordinates = existing_export_coordinates if is_export else existing_operation_coordinates
    
    new_lat, new_lon = lat, lon
    while (new_lat, new_lon) in existing_coordinates:
        new_lat += lat_offset
        new_lon += lon_offset
    
    existing_coordinates.add((new_lat, new_lon))
    return new_lat, new_lon


#------------------ DATASET ------------------
#------------------ IMPORT E MODIFICHE ------------------

df = pd.read_csv('Data_Cleaned.csv', delimiter=';')
df = df.dropna(subset=['importo'])

df['Piazza'] = df['Piazza'].apply(lambda x: str(x).title() if pd.notnull(x) else x)
df['Provincia'] = df['Provincia'].apply(lambda x: str(x).upper() if pd.notnull(x) else x)

df['Piazza'] = df['Piazza'].replace('Salisbury', 'Harare, Zimbabwe')
df['Piazza'] = df['Piazza'].replace('Monza', 'Monza, Italy')

country_mapping = {'urss' : 'russia', 'cecoslovacchia' : 'repubblica ceca',
					'jugoslavia' : 'bosnia', 'gran bretagna' : 'inghilterra',
					'germania orientale' : 'germania', 'germania occidentale' : 'germania', 
					'congo di brazzaville' : 'repubblica del congo'}
df['Export in'] = df['Export in'].replace(country_mapping)

invalid_provinces = ['jugoslavia', 'germania occidentale', 'africa orientale portoghese', 
                     'rhodesia del sud', 'repubblica del dahomey', 'mozambico', 'sudan']
df['Provincia'] = df['Provincia'].apply(lambda x: np.nan if str(x).lower() in invalid_provinces else x)

df['Coordinates'] = df.apply(lambda row: get_location_by_name(row['Piazza'], row['Provincia']) if pd.notnull(row['Piazza']) else None, axis=1)

missing_coordinates_df = df[df['Coordinates'].isnull()]
missing_coordinates_df['Cleaned_Piazza'] = missing_coordinates_df['Piazza'].apply(lambda x: str(x).split('-')[0].strip() if pd.notnull(x) else x)
missing_coordinates_df['Cleaned_Coordinates'] = missing_coordinates_df.apply(lambda row: get_location_by_name(row['Cleaned_Piazza'], row['Provincia']) if pd.notnull(row['Cleaned_Piazza']) else None, axis=1)
df.loc[missing_coordinates_df.index, 'Coordinates'] = missing_coordinates_df['Cleaned_Coordinates'].where(missing_coordinates_df['Cleaned_Coordinates'].notnull(), df.loc[missing_coordinates_df.index, 'Coordinates'])

df['Export_Coordinates'] = df.apply(lambda row: get_export_location({'Capitale(2)': row['Capitale(2)'], 'Export in': row['Export in']}) if pd.notnull(row['Capitale(2)']) else None, axis=1)
missing_export_coordinates_df = df[df['Export_Coordinates'].isnull() & df['Capitale(2)'].notnull() & df['ISO'].notnull()].copy()
missing_export_coordinates_df['New_Export_Coordinates'] = missing_export_coordinates_df.apply(lambda row: get_location_by_name(f"{row['Capitale(2)']}, {row['ISO']}") if pd.notnull(row['Capitale(2)']) and pd.notnull(row['ISO']) else None, axis=1)
df.loc[missing_export_coordinates_df.index, 'Export_Coordinates'] = missing_export_coordinates_df['New_Export_Coordinates'].where(missing_export_coordinates_df['New_Export_Coordinates'].notnull(), df.loc[missing_export_coordinates_df.index, 'Export_Coordinates'])

missing_coordinates_df = df[df['Coordinates'].isnull()]
missing_coordinates_df.to_csv('/Users/filippooberto/Desktop/MedioBanca/missing_coordinates.csv', index=False)
missing_geo_df = df[df['Coordinates'].isnull() | df['Export_Coordinates'].isnull()]
missing_geo_df.to_csv('/Users/filippooberto/Desktop/MedioBanca/missing_geolocations.csv', index=False)

df['Aziende'] = df.groupby('Piazza')['Nominativo'].transform(lambda x: ','.join(x.unique()))
df['Totale Importi'] = df.groupby('Piazza')['importo'].transform('sum')
df['Anno_Fiscale'] = df['anno_fiscale'].str.extract('(\d{4})').astype(float)

#Calcola numero operazioni e importo totale per anno e per piazza
yearly_data = df.groupby(['Piazza', 'Anno_Fiscale'])[['importo']].agg(['count', 'sum']).reset_index()
yearly_data.columns = ['Piazza', 'Anno_Fiscale', 'Operazioni', 'Totale Importi']

#Calcola dettagli operazioni per anno, piazza e importo
operation_details = df.groupby(['Piazza', 'Anno_Fiscale', 'Nominativo'])['importo'].sum().reset_index()
operation_details['Details'] = operation_details.apply(lambda x: f"{int(x['Anno_Fiscale'])}: {x['Nominativo']}; ₤{x['importo']:,}", axis = 1)

operation_details_agg = operation_details.groupby('Piazza')['Details'].apply(list).reset_index()

df = df.merge(operation_details_agg, on='Piazza', how='left')

#Raggruppa operazioni per anno
operations_by_year = df.groupby(['Piazza', 'Anno_Fiscale', 'Nominativo', 'importo']).size().reset_index(name='Count') #'Count' è sempre 1, ma serve per avere tutte le righe

#Formatta i dettagli delle operazioni in un elenco puntato
operations_by_year['Operation_Details'] = operations_by_year.apply(lambda x: f"-{x['Nominativo']}: ₤{x['importo']:,}", axis=1)

operations_by_year_agg = operations_by_year.groupby(['Piazza', 'Anno_Fiscale'])['Operation_Details'].apply(list).reset_index()

#Operazioni per ogni piazza
operations_count = df.groupby('Piazza').size().reset_index(name='Operazioni')
df = df.merge(operations_count, on='Piazza', how='left')

#Calcola dettagli export per anno e azienda
export_details = df.dropna(subset=['Export_Coordinates']).groupby(['Export in', 'Anno_Fiscale', 'Nominativo']).size().reset_index(name='Export_Operations')
export_details['Export_Details'] = export_details.apply(lambda x: f"{x['Nominativo']}", axis=1)
export_yearly_details = export_details.groupby(['Export in', 'Anno_Fiscale'])[['Export_Details', 'Export_Operations']].agg({'Export_Details': list, 'Export_Operations': 'sum'}).reset_index()
export_details_agg = export_yearly_details.groupby('Export in').apply(lambda x: {year: (int(ops), details) for year, ops, details in x[['Anno_Fiscale', 'Export_Operations', 'Export_Details']].to_records(index=False)}).reset_index(name='Yearly_Export_Details')
df = pd.merge(df, export_details_agg, on='Export in', how='left')

#Stampa CSV con linee non localizzate
failed_rows = df[df['Coordinates'].isnull() & df['Piazza'].notnull()]
failed_rows.to_csv('failed_rows.csv', index=False)

failed_export_rows = df[df['Export_Coordinates'].isnull() & df['Capitale(2)'].notnull() & df['Export in'].notnull()]
failed_export_rows.to_csv('failed_export_rows.csv', index=False)

#------------------ PREPARAZIONE LOGO ------------------

with open('mediobanca.jpeg', 'rb') as f:
    image = Image.open(f)
    image = image.convert("RGBA")
    
data = image.getdata()
new_data = []

for item in data:
    if item[0] in list(range(200, 256)):
        new_data.append((255, 255, 0, 0))
    else:
        new_data.append(item)

image.putdata(new_data)
image.save("mediobanca_transparent.png")

#------------------ MAPPA ------------------
#Carica e codifica un JPEG per il logo
logo = base64.b64encode(open('mediobanca_transparent.png', 'rb').read()).decode()

#Creo la mappa e aggiungo marker con cluster
central_coords = [df['Coordinates'].dropna().apply(lambda x: x[0]).mean(), df['Coordinates'].dropna().apply(lambda x: x[1]).mean()]
m = folium.Map(location=central_coords, zoom_start=5)

marker_cluster = MarkerCluster().add_to(m)
export_marker_cluster = MarkerCluster().add_to(m)

logo_html = '''
<div style="position: fixed; 
            top: 10px; 
            left: 10px; 
            width: 140px; 
            height: 140px; 
            z-index:9999;">
    <img src="data:image/jpeg;base64,{}" style="width:160px;height:100px;">
</div>'''.format(logo)

# Aggiungere l'elemento HTML alla mappa
m.get_root().html.add_child(branca.element.Element(logo_html))

existing_coordinates = set()

#Aggiungo ogni coordinata al MarkerCluster con tooltip e popup per Piazza
for _, row in df.drop_duplicates('Piazza').iterrows():
    if row['Coordinates']:
        lat, lon = row['Coordinates']
        existing_coordinates.add((lat, lon))
        lat, lon = add_offset_if_needed(lat, lon, existing_coordinates)
        formatted_importi = "{:,.2f}".format(row['Totale Importi'])
        
        formatted_importi = "{:,.2f}".format(row['Totale Importi'])  # Formatta le cifre per la leggibilità
        if pd.notnull(row['Aziende']):
            aziende_list = '<ul>' + ''.join(f'<li>{azienda}</li>' for azienda in str(row['Aziende']).split(',')) + '</ul>'
        else:
            aziende_list = 'Nessuna azienda disponibile'

        operations_by_year = df[df['Piazza'] == row['Piazza']].groupby('Anno_Fiscale')
        yearly_operations = ""
        for year, operations in operations_by_year:
            operation_list = '<li>' + '</li><li>'.join(
               [f"{op['Nominativo']}: ₤{op['importo']:,}" for _, op in operations.iterrows()]) + '</li>'
            yearly_operations += f"{int(year)}:<ul>{operation_list}</ul><br>"
         

        if isinstance(row['Details'], list):
            operation_details_list = '<br>'.join(row['Details'])
        else:
            operation_details_list = str(row['Details'])
                  
        tooltip_content = f"Piazza: {row['Piazza']}, Operazioni: {row['Operazioni']}"
        popup_content = folium.Popup(f"<div style='max-height: 200px; overflow: auto;'>"
                                     f"<h4><strong>Piazza:</strong> {row['Piazza']}</h4>"
                                     f"<strong>Operazioni:</strong><br>{yearly_operations}<br>"
                                     f"<strong>Totale Importi:</strong> ₤{formatted_importi}<br>"
                                     f"</div>", 
                                     max_width=350)  
        folium.Marker(
            location=(lat, lon), 
            tooltip=tooltip_content, 
            popup=popup_content
        ).add_to(marker_cluster)
        
        existing_coordinates.add(tuple(row['Coordinates']))

#Aggiungo marker per l'export alla mappa
for _, row in df.dropna(subset=['Export_Coordinates']).drop_duplicates('Export in').iterrows():
    lat, lon = row['Export_Coordinates']
    existing_coordinates.add((lat, lon))
    lat, lon = add_offset_if_needed(lat, lon, existing_coordinates)
    
    yearly_export_details = ""
    for year, (ops, details) in row['Yearly_Export_Details'].items():
        detail_list = '<li>' + '</li><li>'.join(details) + '</li>'
        yearly_export_details += f"{int(year)}:<br>Numero operazioni: {ops}<ul>{detail_list}</ul><br>"

    # Aggiunta del tooltip_content qui
    tooltip_content = f"Export: {row['Export in']}, Operazioni: {sum([ops for year, (ops, details) in row['Yearly_Export_Details'].items()])}"

    popup_content = folium.Popup(f"<div style='max-height: 150px; overflow: auto;'>"
                                 f"<strong>Export:</strong> {row['Export in']}<br>"
                                 f"{yearly_export_details}<br>", 
                                 max_width=350)

    folium.Marker(
        location=(lat, lon), 
        tooltip=tooltip_content, 
        popup=popup_content,
        icon=folium.Icon(color='green')
    ).add_to(export_marker_cluster)


#Salvo in HTML
m.save("map.html")