import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px 
import streamlit as st

@st.cache

def load_data():
    df_final = pd.read_csv('df_final_topics.csv')

    df_final['text'] = df_final['text'].astype('str')
    df_final['text_clean'] = df_final['text_clean'].astype('str')
    df_final['text_clean_lemm'] = df_final['text_clean_lemm'].astype('str')

    df_final['date']=df_final['date'].astype('datetime64')
    df_final['date_short']=df_final['date_short'].astype('datetime64')
    df_final['medio']=df_final['medio'].astype('category')
    
    return df_final

df_final = load_data()

indices=pd.DataFrame(df_final.groupby('date_short')['count_DEPU'].mean()).rename(columns={'count_DEPU':'freq_DEPU'})
indices['freq_DEPUC']=pd.DataFrame(df_final.groupby('date_short')['count_DEPUC'].mean()).rename(columns={'count_DEPUC':'freq_DEPUC'})

st.title('Uncertainty indexes')
#st.subheader('EPOC')

fig = px.line(indices, x=indices.index, y="freq_DEPU",title="DEPU frequency of tweets", width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
fig.show()

fig = px.line(indices, x=indices.index, y="freq_DEPUC",title="DEPUC frequency of tweets", width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
fig.show()

