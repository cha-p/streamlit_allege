#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:52:03 2021
Dashboard projet 7
To run : streamlit run P7_dashboard.py
@author: charlottepostel
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

# Import data
data = pd.read_csv("app_test.csv")
model = pickle.load(open("scoring_model_f2.sav", 'rb'))

data_X = data.copy()
data_X = data_X.drop(columns="SK_ID_CURR")

# Features
var_cat = []
for col in data_X.columns:
    if data_X[col].dtypes == object:
        var_cat.append(col)
var_num = data_X.columns
var_num = var_num.drop(var_cat)
name_features = model[0].transformers_[0][1][0].get_feature_names(var_cat)
for i in var_num.values:
    name_features = np.append(name_features, i)

# Lime
x_transformed = pd.DataFrame(model[0].transform(data_X),
                             columns=name_features,
                             index=data_X.index)

lime1 = LimeTabularExplainer(x_transformed,
                             feature_names=name_features,
                             class_names=["Solvable", "Non Solvable"],
                             discretize_continuous=False, random_state=42)


#data['proba'] = model.predict_proba(data_X)[:,1]

# Fonction compte valeur pour pie plot
def count_val(df, input_cl):
    x = df.value_counts()/len(df)*100
    i = x[x<3].index
    try:
        i = i.drop(input_cl)
    except:
        pass
    temp = x[i].sum()
    x = x.drop(i)
    x['Other categories <1%'] = temp
    return x


# Fonction pour faire boxplot/pie plotly pour un client donné
def boxplot(input_d, valeur):
    data_var = data[str(input_d)].copy()
    data_var = data_var.replace(np.nan, 'nan')
    if data[str(input_d)].dtypes == 'O':
        if input_d == 'NAME_EDUCATION_TYPE':
            data_var = data_var.replace('Secondary / secondary special','Secondary special')
        if input_d in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']:
            x = count_val(data_var, valeur)
        else:
            x = data_var.value_counts()/len(data_var)*100
        fig = px.pie(values=x,
                    names=x.index)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update(layout_showlegend=False)
        fig.update_layout(title={'text': "Distribution sur tous les clients",
                                 'y':0.9,
                                 'x':0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)
    elif  set(data[str(input_d)].unique()) == set([0,1]):
        x = data[input_d].value_counts()/len(data)*100
        fig = px.pie(values=x,
                     names=data[input_d].value_counts().index)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update(layout_showlegend=False)
        fig.update_layout(title={'text': "Distribution sur tous les clients",
                                 'y':0.9,
                                 'x':0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Box(y=data[str(input_d)],
                             boxpoints=False,
                             boxmean=True,
                             name=str(input_d)))
        if str(valeur.values[0]) != 'nan':
            fig.add_hline(y=valeur.values[0], line_width=3, line_color="#ff7400")
        fig.update_layout(title={'text': "Distribution sur tous les clients",
                                 'y':0.9,
                                 'x':0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)

# Layout & Navigation panel
st.set_page_config(page_title="Dashboard",
                   page_icon="☮",
                   initial_sidebar_state="expanded")
sb = st.sidebar # add a side bar 
sb.write('# Navigation')
sb.write('###')
rad1 = sb.radio('Pages',('🏠 Accueil', 
                         '🔎 Exploration des données',
                         '📉 Prédiction'))
   

# Déroulement menu en fonction choix
if rad1 == '🏠 Accueil':
    st.title("Dashboard\n ----")
    st.markdown("Cette application interactive est un outil d'aide à la décision pour l'octroi de prêt bancaire.")
    st.markdown("Vous trouverez sur l'onglet de gauche les deux principales fonctionnalités:")
    st.markdown("- Un outil permettant l'exploration des données d'un client sélectionné ainsi que la possibilité de comparer ce client aux autres clients de la banque.")
    st.markdown("- Un outil permettant la visualisation de la probabilité d'être Non solvable, l'interprétation de cette probabilité (Non solvable ou Solvable) ainsi que la description des variables ayant influencées cette prédiction.")

if rad1 == '🔎 Exploration des données':
    label_test = data['SK_ID_CURR'].copy()
    label_test = label_test.sort_values()
    st.title("Exploration des données\n ----")
    colc, colb = st.columns(2)
    with colc:
        input_client = st.selectbox("Veuillez sélectionner l'identifiant du client", label_test)
    #with colb:
        #st.markdown("#")
    #if colb.button('OK'):
    st.markdown("#### Données brutes:")
    st.write(data[data['SK_ID_CURR']==input_client])
    st.write('###')
    st.markdown("#### Exploration des données et Comparaisons avec les autres clients :")
    #st.markdown("Vous trouverez ci-dessous un outil permettant d'obtenir la valeur d'une variable souhaitée ainsi qu'un graphique permettant de visualiser la distribution de cette variable sur tous les clients.")
    st.markdown("####")
    input_d1 = st.selectbox("Veuillez sélectionner une variable", data_X.columns.values)
    col1, col2 = st.columns(2)
    with col1:
        valeur = data.loc[data['SK_ID_CURR']==input_client, input_d1]
        st.markdown("####")
        st.write("Valeur du client : ")
        st.markdown("####")
        st.write('       ', str(valeur.to_numpy()[0]))
    with col2:
        boxplot(input_d1, valeur)
    # with col2:
    #     input_d2 = st.selectbox("Deuxième variable", data_X.columns.values)
    #     valeur2 = data.loc[data['SK_ID_CURR']==input_client, input_d2]
    #     st.write("Valeur du client : ", str(valeur2.to_numpy()[0]))
    #     boxplot(input_d2, valeur2)
    #     st.markdown("####")
    st.markdown("__Notes__ : ")
    st.markdown("La distribution de la variable est représentée par un pie plot ou un boxplot selon la nature de la variable.")
    st.markdown("Pour les variables numériques, le boxplot permet de visualiser la médiane (ligne pleine) et la moyenne (trait pointillé) de la population. Vous pouvez également visualiser la valeur du client sous la forme d'une ligne orange.")

if rad1 == '📉 Prédiction':
    label_test2 = data['SK_ID_CURR'].copy()
    label_test2 = label_test2.sort_values()
    st.title("Prédiction\n ----")
    colc2, colb2 = st.columns(2)
    with colc2:
        input_client2 = st.selectbox("Veuillez sélectionner l'identifiant du client :", label_test2)
    #if colb2.button('Prédiction'):
    col3, col4 = st.columns(2)
    with col3:
        index = data[data['SK_ID_CURR']==input_client2].index
        pred = np.round(model.predict_proba(data_X.iloc[index])[:,1], 2)
        st.write("#### Probabilité d'être non solvable")
        palette = sns.color_palette("Blues_r")
        fig3, ax = plt.subplots(figsize=(4,1))
        ax.set_yticks([1])
        ax.set_yticklabels('')
        ax.set_xlim(0,1)
        ax.barh([1], [pred[0]], height=10, color=palette)
        ax.axvline(0.48,  color='r', ls='--', alpha=0.9)
        for i, v in enumerate([pred[0]]):
            ax.text(1.04, 0.1, str(v), color='black', size=15, verticalalignment='center')
            #ax.set_title("Probabilité d'être non solvable", size=20, pad=20)
            st.write(fig3)
            st.write('###')
    with col4:
        st.write('####    Prédiction :')
        st.write('#####')
        st.write('#####')
        from annotated_text import annotated_text
        if pred[0]> 0.48:
            annotated_text(("    ", "", ""), ("Solvable", "", ""),
                           ( " /", "", ""),
                           ("Non solvable", "", "#f90036"))
        else:
            annotated_text(("    ", "", ""), ("Solvable", "", "#5cc07f"),
                           (" /", "", ""),
                           ("Non solvable", "", ""))
        
    st.markdown("#### Principales variables influençant la prédiction :")
    st.markdown("Graphique représentant l'influence des variables sur la prédiction en faveur de la classe Non solvable (rouge) et en faveur de la classe Solvable (vert)")
    index = data[data['SK_ID_CURR']==input_client2].index
    exp = lime1.explain_instance(x_transformed.iloc[index.values[0]], 
                                     model[1].predict_proba,
                                     num_samples=1000)
    ex = pd.DataFrame(exp.as_list(), columns=['var', 'valeur'])
    ex["Color"] = np.where(ex['valeur']<0, 'green', 'red')
    fig5 = go.Figure()
    fig5.add_trace(
            go.Bar(name='Valeur',
                   y=ex['var'],
                   x=ex['valeur'],
                   marker_color=ex['Color'],
                   orientation='h'))
    fig5.update_layout(barmode='stack',
                  yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig5, use_container_width=True)
        
    # plot variables influence sur non solvable
    # ex["variables"] = ex['var'].apply(lambda x: '_'.join(x.split('_')[:-1]) if any(i in x for i in var_cat) else 0)
    # ex.loc[ex['variables']==0, 'variables'] = ex['var'].copy()
    # st.markdown("#### Exploration des variables influençant la prédiction en faveur de la classe Non solvable")
    # st.markdown("#####")
    # col5, col6 = st.columns(2)
    # with col5:
    #     input_d3 = st.selectbox("Veuillez sélectionner la variable souhaitée", ex.loc[ex['valeur']>0,'variables'].values)
    #     valeur3 = data.loc[data['SK_ID_CURR']==input_client2, input_d3]
    #     st.write("Valeur du client: ", str(valeur3.to_numpy()[0]))
    #     boxplot(input_d3, valeur3)
    # with col6:
    #     input_d4 = st.selectbox("Autre variable", ex.loc[ex['valeur']>0,'variables'].values)
    #     valeur4 = data.loc[data['SK_ID_CURR']==input_client2, input_d4]
    #     st.write("Valeur du client: ", str(valeur4.to_numpy()[0]))
    #     boxplot(input_d4, valeur4)
        
    # # plot variables influence sur solvable
    # st.markdown("#### Exploration des variables influençant la prédiction en faveur de la classe Solvable")
    # st.markdown("#####")
    # col7, col8 = st.columns(2)
    # with col7:
    #     input_d5 = st.selectbox("Veuillez sélectionner la variable souhaitée :", sorted(ex.loc[ex['valeur']<0,'variables'].values))
    #     valeur5 = data.loc[data['SK_ID_CURR']==input_client2, input_d5]
    #     st.write("Valeur du client : ", str(valeur5.to_numpy()[0]))
    #     boxplot(input_d5, valeur5)
    # with col8:
    #     input_d6 = st.selectbox("Autre variable :", sorted(ex.loc[ex['valeur']<0,'variables'].values))
    #     valeur6 = data.loc[data['SK_ID_CURR']==input_client2, input_d6]
    #     st.write("Valeur du client : ", str(valeur6.to_numpy()[0]))
    #     boxplot(input_d6, valeur6)
