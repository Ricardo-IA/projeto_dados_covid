#!/usr/bin/env python
# coding: utf-8

# # Projeto COVID-19
# ## Digital Innovation One
# Primeiro vaomos impotar algumas das bibliotecas necessárias para nosso projeto

# In[39]:


import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# In[40]:


# vamos importar os dados para o projeto
url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'


# In[41]:


df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])
df


# In[42]:


#Conferir os tipos de cada coluna
df.dtypes


# nomes de colunas não deve ter letras maiúsculas e nem nem caracteres especiais. Vamos implementar uma função para fazer a limpeza dos nomes dessas colunas

# In[43]:


import re
def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()


# In[44]:


corrige_colunas("LP/Edtu") #testando


# In[45]:


#Vamos corrigir todas as colunas do df
df.columns = [corrige_colunas(col) for col in df.columns]


# In[46]:


df


# # Brasil
# ## Vamos selecionar apenas os dados do Brasil para investigar

# In[47]:


#contagem dos casos nos paises
df.countryregion.value_counts()


# In[48]:


#lista de todos os pais
df.countryregion.unique()


# In[49]:


df.loc[df.countryregion == 'Brazil']


# In[50]:


Brasil =  df.loc[
    (df.countryregion == 'Brazil') & (df.confirmed > 0)
]
Brasil


# ## Casos Confirmados

# In[51]:


#Gráfico da evolução de casos confirmados
px.line(Brasil, 'observationdate', 'confirmed', 
        labels={'observationdate':'Data', 'confirmed':'Número de casos confirmados'},
       title='Casos confirmados no Brasil')


# # Novos Casos por dia

# In[52]:


# Técnica de programação funional
Brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else Brasil['confirmed'].iloc[x] - Brasil['confirmed'].iloc[x-1],
    np.arange(Brasil.shape[0])
))


# In[53]:


# Visualizando
px.line(Brasil, x='observationdate', y='novoscasos', title='Novos casos por dia',
       labels={'observationdate': 'Data', 'novoscasos': 'Novos casos'})


# ## Mortes

# In[54]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=Brasil.observationdate, y=Brasil.deaths, name='Mortes', mode='lines+markers',
              line=dict(color='red'))
)
#Edita o layout
fig.update_layout(title='Mortes por COVID-19 no Brasil',
                   xaxis_title='Data',
                   yaxis_title='Número de mortes')
fig.show()


# ## Taxa de crescimento
# taxa de crescimento =(presente/passado)**(1/n))-1

# In[55]:


def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # Se data_inicio for None, define como a primeira data disponível no dataset
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
    
    # Define os valores de presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    
    # Define o número de pontos no tempo q vamos avaliar
    n = (data_fim - data_inicio).days
    
    # Calcula a taxa
    taxa = (presente/passado)**(1/n) - 1

    return taxa*100


# In[56]:


#taxa de crescimento médio de COVID no Brasil em todo o periodo
taxa_crescimento(Brasil, 'confirmed')


# In[57]:


def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    data_fim = data.observationdate.max()
    n = (data_fim - data_inicio).days
    #taxa calculada de um dia para o outro
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1,n+1)
    ))
    return np.array(taxas)*100


# In[58]:


tx_dia = taxa_crescimento_diaria(Brasil, 'confirmed')


# In[59]:


tx_dia


# In[60]:


primeiro_dia = Brasil.observationdate.loc[Brasil.confirmed > 0].min()
px.line(x=pd.date_range(primeiro_dia, Brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil',
       labels={'y':'Taxa de crescimento', 'x':'Data'})


# ## Predições
# construindo um modelo de séries temporais para prever os novos casos. Antes analisemos a série tempora

# In[61]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# In[62]:


novoscasos = Brasil.novoscasos
novoscasos.index = Brasil.observationdate

res = seasonal_decompose(novoscasos)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.scatter(novoscasos.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()


# ## CASOS CONFIRMADOS
# 

# In[63]:


confirmados = Brasil.confirmed
confirmados.index = Brasil.observationdate


# In[64]:


res2 = seasonal_decompose(confirmados)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
ax1.plot(res2.observed)
ax2.plot(res2.trend)
ax3.plot(res2.seasonal)
ax4.scatter(confirmados.index, res2.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()


# ## Auto Arima
# Antever o numero de casos confirmados

# In[65]:


get_ipython().system('pip install pmdarima')


# In[66]:


from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)


# In[67]:


fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observed'
))

fig.add_trace(go.Scatter(x=confirmados.index, y = modelo.predict_in_sample(), name='Predicted'))

fig.add_trace(go.Scatter(x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast'))

fig.update_layout(title='Previsão de casos confirmados para os próximos 30 dias',
                 yaxis_title='Casos confirmados', xaxis_title='Data')
fig.show()


# ## Modelo de Crescimento
# Forecasting com Facebook Prophet

# In[68]:


get_ipython().system('conda install -c conda-forge fbprophet -y')


# In[69]:


from fbprophet import Prophet


# In[ ]:


# Preprocessamentos
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# renomeia colunas
train.rename(columns={"observationdate":"ds","confirmed":"y"},inplace=True)
test.rename(columns={"observationdate":"ds","confirmed":"y"},inplace=True)
test = test.set_index("ds")
test = test['y']

profeta = Prophet(growth="logistic", changepoints=['2020-03-21', '2020-03-30', '2020-04-25', '2020-05-03', '2020-05-10'])

#pop = 1000000
pop = 211463256 #https://www.ibge.gov.br/apps/populacao/projecao/box_popclock.php
train['cap'] = pop

# Treina o modelo
profeta.fit(train)

# Construindo previsões para o futuro
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast =  profeta.predict(future_dates)


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




