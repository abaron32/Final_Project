import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.io as pio 
import streamlit as st

st.set_page_config(layout="wide")

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








st.title('Analysis of tweets from Uruguayan media press')

st.subheader('Introduction')

pio.templates.default = "plotly_white" # template for plotly express plots

col1, col2 = st.columns(2)

fig1 = px.histogram(df_final, x="medio",title="Number of tweets per medio", width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(df_final, x="medio",title="Number of tweets per medio", width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col2.plotly_chart(fig2, use_container_width=True)


###############################################################################################################333


st.subheader('Uncertainty indexes')

col1, col2 = st.columns(2)

fig1 = px.line(indices, x=indices.index, y="freq_DEPU",title="DEPU frequency of tweets", width=1000, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
#fig.show()
col1.plotly_chart(fig1, use_container_width=True)


fig2 = px.line(indices, x=indices.index, y="freq_DEPUC",title="DEPUC frequency of tweets", width=1000, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
#fig.show()
col2.plotly_chart(fig2, use_container_width=True)
