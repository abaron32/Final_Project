import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.io as pio 
import streamlit as st

st.set_page_config(layout="wide")

@st.cache

def load_data():
    df_final = pd.read_csv('./data/processed/df_final_topics.csv')

    df_final['text'] = df_final['text'].astype('str')
    df_final['text_clean'] = df_final['text_clean'].astype('str')
    df_final['text_clean_lemm'] = df_final['text_clean_lemm'].astype('str')

    df_final['date']=df_final['date'].astype('datetime64')
    df_final['date_short']=df_final['date_short'].astype('datetime64')
    df_final['medio']=df_final['medio'].astype('category')
    
    return df_final

df_final = load_data()

#Intro visualizations

df_plot_day=(df_final
        .assign(day=df_final['date'].dt.day_of_week)
        .groupby(['medio','day'])
        .agg(count=('medio','count'))
        .assign(rate=lambda df_interim: df_interim['count']/df_interim.groupby('medio')['count'].sum())
        .reset_index())


# Uncertainty indices
indices=pd.DataFrame(df_final.groupby('date_short')['count_DEPU'].mean()).rename(columns={'count_DEPU':'freq_DEPU'})
indices['freq_DEPUC']=pd.DataFrame(df_final.groupby('date_short')['count_DEPUC'].mean()).rename(columns={'count_DEPUC':'freq_DEPUC'})








st.title('Analysis of tweets from Uruguayan media press')

st.subheader('Introduction')

st.write('''In this project we build economic policy uncertainty indexes (following Becerra et al (2020) and Baker 
et al (2016)) and analyze emotions, sentiments and topics using tweets from the media press in Uruguay between March 
2022 and August 2022. In order to make good policy decisions, policymakers need timeliness and frequent information, 
but many economic indicators are published with considerable lags. Natural language processing techniques allows us to 
summarize information from the social media Twitter and contribute to the decision-making process with timeliness 
indicators.''')


pio.templates.default = "plotly_white" # template for plotly express plots

col1, col2 = st.columns(2)

fig1 = px.histogram(df_final, x="medio",title="Number of tweets per medio", width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(df_plot_day, x="day", y="rate", color='medio', title='Frequency of tweets per media by day', width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col2.plotly_chart(fig2, use_container_width=True)

###############################################################################################################333


st.subheader('Uncertainty indexes')


st.write('Aca capaz podemos hablar un poco de como construimos los indicadores de incertidumbre y que usamos los terminos de la tabla de abajo')

# Tabla con terminos
tabla_terminos = pd.read_csv('tabla_terminos.csv', index_col=0)
st.dataframe(tabla_terminos)


# Graficos
col1, col2 = st.columns(2)

fig1 = px.line(indices, x=indices.index, y="freq_DEPU",title="DEPU frequency of tweets", width=1000, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col1.plotly_chart(fig1, use_container_width=True)


fig2 = px.line(indices, x=indices.index, y="freq_DEPUC",title="DEPUC frequency of tweets", width=1000, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col2.plotly_chart(fig2, use_container_width=True)


st.subheader('Sentiments and emotions')


st.subheader('Topics')



##################################################################################################################3


st.subheader('References')

st.write('Baker, S.R., Bloom, N. and Davis, S.J. (2016). Measuring economic policy uncertainty. The Quarterly Journal of Economics, Volume 131, Issue 4.')
st.write('Becerra, J.S. and Stagner A. (2020). Twitter-based economic policy uncertainty index for Chile. Working Paper 883, Banco Central de Chile.')