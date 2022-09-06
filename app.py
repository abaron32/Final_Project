import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import streamlit as st

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)


## Read all dfs

# df final with topics 

def load_data():
    df_final = pd.read_csv('./data/processed/df_final_topics.csv')

    df_final['text'] = df_final['text'].astype('str')
    df_final['text_clean'] = df_final['text_clean'].astype('str')
    df_final['text_clean_lemm'] = df_final['text_clean_lemm'].astype('str')

    df_final['date']=df_final['date'].astype('datetime64')
    df_final['date_short']=df_final['date_short'].astype('datetime64')
    df_final['medio']=df_final['medio'].astype('category')
    df_final['Month']=df_final['date'].dt.month
    return df_final

df_final = load_data()

# sentiment prediction
sent_pred = pd.read_csv('./data/processed/Sent_pred.csv')


# df with topics for all tweets
topics_words = pd.read_csv('./data/processed/Tabla_1.csv')

# words table
tabla_terminos = pd.read_csv('./data/processed/tabla_terminos.csv', index_col=0)










st.title('Analysis of tweets from Uruguayan media press')

st.subheader('Introduction')

st.write('''In this project we build economic policy uncertainty indexes (following Becerra et al (2020) and Baker 
et al (2016)) and analyze sentiments and topics using tweets from the media press in Uruguay between March 
2022 and August 2022. In order to make good policy decisions, policymakers need timeliness and frequent information, 
but many economic indicators are published with considerable lags. Natural language processing techniques allows us to 
summarize information from the social media Twitter and contribute to the decision-making process with timeliness 
indicators.''')

# df to plot frequency of tweets by day
df_plot_day=(df_final
        .assign(day=df_final['date'].dt.day_of_week)
        .groupby(['medio','day'])
        .agg(count=('medio','count'))
        .assign(rate=lambda df_interim: df_interim['count']/df_interim.groupby('medio')['count'].sum())
        .reset_index())

pio.templates.default = "plotly_white" # template for plotly express plots

col1, col2 = st.columns(2)

fig1 = px.histogram(df_final, x="medio",title="Number of tweets per medio", width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(df_plot_day, x="day", y="rate", color='medio', title='Frequency of tweets per media by day', width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col2.plotly_chart(fig2, use_container_width=True)

###############################################################################################################


st.subheader('Uncertainty indexes')

# Tabla con terminos
st.dataframe(tabla_terminos)

# Uncertainty indices
indices_day=pd.DataFrame(df_final.groupby('date_short')['count_DEPU'].mean()).rename(columns={'count_DEPU':'freq_DEPU'})
indices_day['freq_DEPUC']=pd.DataFrame(df_final.groupby('date_short')['count_DEPUC'].mean()).rename(columns={'count_DEPUC':'freq_DEPUC'})



# Plots
col1, col2 = st.columns(2)

fig1 = px.line(indices_day, x=indices_day.index, y="freq_DEPU",title="DEPU frequency of tweets", labels={
                     "date_short": "Date",
                     "freq_DEPU": "DEPU"
                 }, width=1000, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)


col1.plotly_chart(fig1, use_container_width=True)


fig2 = px.line(indices_day, x=indices_day.index, y="freq_DEPUC",title="DEPUC frequency of tweets", labels={
                     "date_short": "Date",
                     "freq_DEPUC": "DEPUC"
                 }, width=1000, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col2.plotly_chart(fig2, use_container_width=True)


###############################################################################################################

st.subheader('Sentiment analysis')

# Add colum with main sentiment
df_final['Sentiment']=sent_pred['Sentiment'] 

# Frequency of sentiments by day
df_final['Neutral'] = ''
df_final['Neutral'] = ['1' if x == 'NEU' else '0' for x in df_final['Sentiment']]

df_final['Negative'] = ''
df_final['Negative'] = ['1' if x == 'NEG' else '0' for x in df_final['Sentiment']]

df_final['Positive'] = ''
df_final['Positive'] = ['1' if x == 'POS' else '0' for x in df_final['Sentiment']]

for var in ['Neutral', 'Negative', 'Positive']:
  df_final[var] = df_final[var].astype(int)


# DEPU
sent_depu=pd.DataFrame(df_final[df_final['count_DEPU']==1].groupby('date_short')['Neutral'].mean()).rename(columns={'Neutral':'freq_neutral'})
sent_depu['freq_negative']=pd.DataFrame(df_final[df_final['count_DEPU']==1].groupby('date_short')['Negative'].mean()).rename(columns={'Negative':'freq_negative'})
sent_depu['freq_positive']=pd.DataFrame(df_final[df_final['count_DEPU']==1].groupby('date_short')['Positive'].mean()).rename(columns={'Positive':'freq_positive'})


# DEPUC
sent_depuc=pd.DataFrame(df_final[df_final['count_DEPUC']==1].groupby('date_short')['Neutral'].mean()).rename(columns={'Neutral':'freq_neutral'})
sent_depuc['freq_negative']=pd.DataFrame(df_final[df_final['count_DEPUC']==1].groupby('date_short')['Negative'].mean()).rename(columns={'Negative':'freq_negative'})
sent_depuc['freq_positive']=pd.DataFrame(df_final[df_final['count_DEPUC']==1].groupby('date_short')['Positive'].mean()).rename(columns={'Positive':'freq_positive'})



# Plots
col1, col2 = st.columns(2)

fig1 = go.Figure(data=[
    go.Bar(name='Negative', x=sent_depu.index, y=sent_depu['freq_negative']),
    go.Bar(name='Positive', x=sent_depu.index, y=sent_depu['freq_positive']),
    go.Bar(name='Neutral', x=sent_depu.index, y=sent_depu['freq_neutral'])
])

# Change the bar mode
fig1.update_layout(barmode='stack', title_text='Sentiment of tweets over time DEPU')
col1.plotly_chart(fig1, use_container_width=True)


fig2 = go.Figure(data=[
    go.Bar(name='Negative', x=sent_depuc.index, y=sent_depuc['freq_negative']),
    go.Bar(name='Positive', x=sent_depuc.index, y=sent_depuc['freq_positive']),
    go.Bar(name='Neutral', x=sent_depuc.index, y=sent_depuc['freq_neutral'])
])

# Change the bar mode
fig2.update_layout(barmode='stack', title_text='Sentiment of tweets over time DEPUC')
col2.plotly_chart(fig2, use_container_width=True)



st.subheader('Topics')

# DEPU's tweets without considering topic -1
df_final_depu=df_final[(df_final['count_DEPU']==1) & (df_final['Topic']!=-1)]

# Top 8 DEPU topics
Top_8_DEPU=pd.DataFrame(df_final_depu['Topic'].value_counts(normalize=True)).index.values[:8]


# DEPUC's tweets without considering topic -1
df_final_depuc=df_final[(df_final['count_DEPUC']==1) & (df_final['Topic']!=-1)]

# Top 8 DEPUC topics
Top_8_DEPUC=pd.DataFrame(df_final_depuc['Topic'].value_counts(normalize=True)).index.values[:8]



# Words in topics

# Plots
col1, col2 = st.columns(2)

topics_words[topics_words['topic'].isin(Top_8_DEPU)].reset_index(drop=True)
fig1 = make_subplots(rows=2, cols=4,
                    subplot_titles=tuple(['Topic '+str(i) for i in Top_8_DEPU]))

for i,t in enumerate(Top_8_DEPU):
  fig1.add_trace(
      go.Bar(x=topics_words[topics_words['topic']==t].word, y=topics_words[topics_words['topic']==t].prob),
      row=(i//4)+1, col=(i%4)+1
      
      )
  
fig1.update_layout(height=800, width=1000, title_text="Words in topics DEPU",showlegend=False)
col1.plotly_chart(fig1, use_container_width=True)

topics_words[topics_words['topic'].isin(Top_8_DEPUC)].reset_index(drop=True)
fig2 = make_subplots(rows=2, cols=4,
                    subplot_titles=tuple(['Topic '+str(i) for i in Top_8_DEPUC]))

for i,t in enumerate(Top_8_DEPUC):
  fig2.add_trace(
      go.Bar(x=topics_words[topics_words['topic']==t].word, y=topics_words[topics_words['topic']==t].prob),
      row=(i//4)+1, col=(i%4)+1
      
      )
  
fig2.update_layout(height=800, width=1000, title_text="Words in topics DEPUC",showlegend=False)
col2.plotly_chart(fig2, use_container_width=True)


# Heatmaps: frequency of topics by medio

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

depu_crosstab=pd.crosstab(df_final_depu[df_final_depu['Topic'].isin(Top_8_DEPU)]['medio'],df_final_depu[df_final_depu['Topic'].isin(Top_8_DEPU)]['Topic'],normalize='index')
depuc_crosstab=pd.crosstab(df_final_depuc[df_final_depuc['Topic'].isin(Top_8_DEPU)]['medio'],df_final_depuc[df_final_depuc['Topic'].isin(Top_8_DEPUC)]['Topic'],normalize='index')

# Plots

col1, col2 = st.columns(2)

fig1 = go.Figure(data=go.Heatmap(df_to_plotly(depu_crosstab)))
fig1.update_xaxes(type='category')
fig1.update_layout(title_text='Heatmap topics per medio DEPU')
col1.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure(data=go.Heatmap(df_to_plotly(depuc_crosstab)))
fig2.update_xaxes(type='category')
fig2.update_layout(title_text='Heatmap topics per medio DEPU')
col2.plotly_chart(fig2, use_container_width=True)


# Topics over time

# DEPU

# Create dummies for topics in depu dataframe

topic_depu_dummies= pd.get_dummies(df_final_depu["Topic"])

# df with dummies of topics
listDF = []
for i in Top_8_DEPU:  
    listDF.append(topic_depu_dummies[i])
listDF=pd.DataFrame(listDF)
listDF=listDF.T

# add dummies to df depu
df_final_depu=pd.concat([df_final_depu,listDF], axis=1)

# create dataframe with topics frequency by day
freq_topics_depu=pd.DataFrame()
for i in Top_8_DEPU:
  new_col=('Topic_'+str(i))
  freq_topics_depu[new_col]=df_final_depu.groupby('date_short')[i].sum()

# centered moving average of topics frequency
freq_topics_depu_mavg=freq_topics_depu.rolling(7,center=True).mean()



# DEPUC

# Create dummies for topics in depuc dataframe

topic_depuc_dummies= pd.get_dummies(df_final_depuc["Topic"])

# df with dummies of topics
listDF = []
for i in Top_8_DEPUC:  
    listDF.append(topic_depuc_dummies[i])
listDF=pd.DataFrame(listDF)
listDF=listDF.T

# add dummies to df depu
df_final_depuc=pd.concat([df_final_depuc,listDF], axis=1)

# create dataframe with topics frequency by day
freq_topics_depuc=pd.DataFrame()
for i in Top_8_DEPUC:
  new_col=('Topic_'+str(i))
  freq_topics_depuc[new_col]=df_final_depuc.groupby('date_short')[i].sum()

# centered moving average of topics frequency
freq_topics_depuc_mavg=freq_topics_depuc.rolling(7,center=True).mean()



# Plots

col1, col2 = st.columns(2)

fig1 = px.line(freq_topics_depu_mavg,title="Topics over time DEPU", labels={
                     "date_short": "Date",
                     "value": "Count tweets"
                 }, width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(freq_topics_depuc_mavg,title="Topics over time DEPUC", labels={
                     "date_short": "Date",
                     "value": "Count tweets"
                 },width=800, height=400, color_discrete_sequence=px.colors.qualitative.Dark24)
col2.plotly_chart(fig2, use_container_width=True)



##############################################################################################################

st.subheader('Validation of uncertainty index')

## DEPU and DEPUC with standard deviation of ER expectation

# Standard deviation of exchange rate expectation for next 12 months -- proxy of uncertainty
Std_TC_12m=pd.DataFrame({'Month':['3','4','5','6','7','8'],'Std_Dv':[1.55,1.55,1.46,1.31,1.38,1.09]})
Std_TC_12m['Month']=Std_TC_12m['Month'].astype('int64')
Std_TC_12m.set_index('Month',inplace=True)

# DEPU and DEPUC by month
indices_month=pd.DataFrame(df_final.groupby('Month')['count_DEPU'].agg(['mean','std'])).rename(columns={'mean':'freq_DEPU','std':'std_DEPU'})
indices_month[['freq_DEPUC','std_DEPUC']]=pd.DataFrame(df_final.groupby('Month')['count_DEPUC'].agg(['mean','std'])).rename(columns={'mean':'freq_DEPUC','count':'std_DEPUC'})

# Standarized frequency
indices_month['std_freq_DEPU']=indices_month['freq_DEPU']/indices_month['std_DEPU']
indices_month['std_freq_DEPUC']=indices_month['freq_DEPUC']/indices_month['std_DEPUC']

# Merge DEPU-DEPUC with std dev of ER
indices_month=indices_month.merge(Std_TC_12m,left_index=True,right_index=True)

# Plot

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Line(x=indices_month.index, y=indices_month['std_freq_DEPU'], name="DEPU"),
    secondary_y=False,
)

# Add traces
fig.add_trace(
    go.Line(x=indices_month.index, y=indices_month['std_freq_DEPUC'], name="DEPUC"),
    secondary_y=False,
)

fig.add_trace(
    go.Line(x=indices_month.index, y=indices_month['Std_Dv'], name="Std Dev ER 12m"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Indexes and standard deviation of exchange rate expectation over time"
)

# Set x-axis title
fig.update_xaxes(title_text="Month")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Index</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Std Dev ER 12m</b>", secondary_y=True)

st.plotly_chart(fig,  use_container_width=True)


## Topics and indexes

# DEPU

# Add column month to dataframe with indexes frequency
freq_topics_depu['Month']=freq_topics_depu.index.month

freq_topics_depu.reset_index(inplace=True,drop=True)
freq_topics_depu.set_index('Month',inplace=True)

# Monthly dataframe
freq_topics_depu=freq_topics_depu.groupby('Month').sum()

# Merge with exchange rate expectation
freq_topics_depu=freq_topics_depu.merge(Std_TC_12m,left_index=True,right_index=True)


# DEPUC

# Add column month to dataframe with indexes frequency
freq_topics_depuc['Month']=freq_topics_depuc.index.month

freq_topics_depuc.reset_index(inplace=True,drop=True)
freq_topics_depuc.set_index('Month',inplace=True)

# Monthly dataframe
freq_topics_depuc=freq_topics_depuc.groupby('Month').sum()

# Merge with exchange rate expectation
freq_topics_depuc=freq_topics_depuc.merge(Std_TC_12m,left_index=True,right_index=True)

# Plots

col1, col2 = st.columns(2)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

for i,t in enumerate(Top_8_DEPU):
  fig1.add_trace(
      go.Line(x=freq_topics_depu.index, y=freq_topics_depu['Topic_'+str(t)],
              name='Topic_'+str(t)),
      secondary_y=False
      )

fig1.add_trace(
    go.Line(x=freq_topics_depu.index, y=freq_topics_depu['Std_Dv'], name="Std dev ER 12m"),
    secondary_y=True,
)

# Add figure title
fig1.update_layout(
    title_text="Topics and standard deviation exchange rate expectation over time DEPU"
)
# Set x-axis title
fig1.update_xaxes(title_text="Month")

# Set y-axes titles
fig1.update_yaxes(title_text="<b>Topics</b>", secondary_y=False)
fig1.update_yaxes(title_text="<b>Std dev ER 12m</b>", secondary_y=True)

col1.plotly_chart(fig1, use_container_width=True)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])

for i,t in enumerate(Top_8_DEPUC):
  fig2.add_trace(
      go.Line(x=freq_topics_depuc.index, y=freq_topics_depuc['Topic_'+str(t)],
              name='Topic_'+str(t)),
      secondary_y=False
      )

fig2.add_trace(
    go.Line(x=freq_topics_depuc.index, y=freq_topics_depuc['Std_Dv'], name="Std dev ER 12m"),
    secondary_y=True,
)

# Add figure title
fig2.update_layout(
    title_text="Topics and standard deviation exchange rate expectation over time DEPUC"
)
# Set x-axis title
fig2.update_xaxes(title_text="Month")

# Set y-axes titles
fig2.update_yaxes(title_text="<b>Topics</b>", secondary_y=False)
fig2.update_yaxes(title_text="<b>Std dev ER 12m</b>", secondary_y=True)

col2.plotly_chart(fig2, use_container_width=True)


##################################################################################################################3


st.subheader('References')

st.write('Baker, S.R., Bloom, N. and Davis, S.J. (2016). *Measuring economic policy uncertainty*. The Quarterly Journal of Economics, Volume 131, Issue 4.')

st.write('Becerra, J.S. and Stagner A. (2020). *Twitter-based economic policy uncertainty index for Chile*. Working Paper 883, Banco Central de Chile.')

st.write('Crocco, N., Dizioli,G.,Herrera, S. (2019). *Construcción de un indicador de incertidumbre económica en base a las noticias de prensa*. Posgrade dissertation. Facultad de Ingeniería. Universidad de la República.')

st.write('Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203. 05794.')

st.write('Pérez, J. M., Furman, D. A., Alemany, L. A., & Luque, F. (2021). *RoBERTuito: a pre-trained language model for social media text in Spanish*. arXiv preprint arXiv:2111. 09453.')

st.write('Pérez, J. M., Giudici, J. C., & Luque, F. (2021). pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks. arXiv [cs.CL]. Ανακτήθηκε από http://arxiv.org/abs/2106.09462')