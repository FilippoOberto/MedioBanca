#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:31:05 2023

@author: filippooberto
"""

import spacy
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import community as community_louvain

nlp = spacy.load('it_core_news_lg')

#-------------------- FUNZIONI --------------------

#Funzione per estrarre entità
def extract_entities(text, nlp):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON']]

#Funzione per creare il grafo
def create_graph(data, nlp):
    G = nx.Graph()
    for idx, row in data.iterrows():
        nominativo = row['Nominativo']
        if nominativo not in G:
            G.add_node(nominativo)
        G.add_edge(nominativo, "Mediobanca")

        for column in ['Utilizzo', 'Garanzie', 'Soci', 'Attività', 'Scopo']:
            text = row.get(column)
            if pd.notna(text):
                for entity in extract_entities(text, nlp):
                    if entity not in G: 
                        G.add_node(entity)
                    G.add_edge(nominativo, entity)
    return G

#Funzione per visualizzare grafo con cluster
def plot_static_graph(G, partition, pos, dimensione_nodi, file_path):
    plt.figure(figsize=(100, 100))
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, node_size=dimensione_nodi, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5, edgelist=[e for e in G.edges() if "Mediobanca" not in e])
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')

    plt.savefig(file_path, format='png', bbox_inches='tight')
    plt.close()
    
#Funzione per visualizzare grafo dinamico 
def plot_dynamic_graph(G, partition, pos, html_file_path='dynamic_graph.html'):
    # Filtra gli archi
    filtered_edges = [e for e in G.edges() if "Mediobanca" not in e]

    # Imposta la dimensione massima dei nodi
    max_node_size = 20
    node_size = [min(G.degree(n) + 5, max_node_size) for n in G.nodes]

    # Crea i bordi
    edge_x, edge_y = [], []
    for edge in filtered_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Crea la figura per i bordi
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Crea la figura per i nodi
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_size,  # Usa la variabile node_size qui
            color=[partition[node] for node in G.nodes],
            colorbar=dict(
                thickness=15,
                title='Community',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    node_trace.text = list(G.nodes())

    # Impostazioni di layout
    layout = go.Layout(
        title='<br>Network graph',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Grafo delle relazioni",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=3000,
        height=3000,
        dragmode='zoom')

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.update_layout(autosize=True)
    fig.write_html(html_file_path)

#-------------------- CARICAMENTO DATASET --------------------
#-------------------- SETTING PARAMETRI COMUNI --------------------
    
#Caricamento Dati
data = pd.read_csv('Data_Cleaned.csv', sep=";")
data = data.dropna(subset=['Nominativo'])

G = create_graph(data, nlp)
partition = community_louvain.best_partition(G)

#Parametri comuni
k_value = 0.15
node_size_multiplier = 10

pos = nx.kamada_kawai_layout(G)

gradi = dict(G.degree())
dimensione_nodi = [500 if nodo == "Mediobanca" else gradi[nodo] * 10 for nodo in G.nodes()]

#Stampo i Grafi
plot_static_graph(G, partition, pos, dimensione_nodi, file_path='8.png')

plot_dynamic_graph(G, partition, pos)


#Esportazione dei nodi in CSV
df_nodi = pd.DataFrame(G.nodes(), columns=['Nodo'])
df_nodi.to_csv('nodi_export.csv', index=False)

# Esportazione degli archi in CSV
df_archi = pd.DataFrame(G.edges(), columns=['Nodo1', 'Nodo2'])
df_archi.to_csv('archi_export.csv', index=False)


                    