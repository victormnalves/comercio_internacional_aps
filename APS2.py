#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from linearmodels import PanelOLS


# In[11]:


#Carregando os dados da base itpd
itpd = pd.read_stata(r'com_int_dados_aps2/itpd.dta')
itpd.head()


# In[12]:


#Transformando variáveis

itpd['trade'] = itpd['trade'] / 10**9

#Filtrando exportadores e importadores
itpd = itpd[itpd['exporter_iso3'] != itpd['importer_iso3']]
itpd.head()


# In[13]:


# Criando base para exportações
itpd_e = itpd[['exporter_m49','broad_sector','year','trade']] # selecionando variáveis de interesse
itpd_e.rename(columns = {'exporter_m49':'Country', 'broad_sector':'Indústria','trade':'Export'}, inplace = True) # renomeando colunas
itpd_e


# In[14]:


# a) 
export_per_sector = itpd_e.groupby(['Country','Indústria','year']).agg(Export=('Export', 'sum'))
export_per_sector


# In[15]:


# b) 
export_per_sector['Exportações Ano'] = export_per_sector.groupby(['Country', 'year'])['Export'].transform('sum')
export_per_sector


# In[16]:


export_per_sector['Vantagem Absoluta'] = (export_per_sector['Export']/export_per_sector['Exportações Ano'])
export_per_sector


# In[17]:


# c) 
# EXit representa as exportações totais da industria i no ano t e EXt as exportacoes
# totais no ano t.

export_per_sector['Exportações Industria Ano'] = export_per_sector.groupby(['Indústria', 'year'])['Export'].transform('sum')
export_per_sector


# In[18]:


export_per_sector['Vantagem Comparativa'] = export_per_sector['Vantagem Absoluta']/((export_per_sector['Exportações Industria Ano']/export_per_sector['Exportações Ano']))
export_per_sector


# In[19]:


export_per_sector_2016 = export_per_sector.xs(2016, level='year')
export_per_sector_2016


# In[20]:


# Removendo outliers
export_per_sector_2016 = export_per_sector_2016[(export_per_sector_2016['Vantagem Comparativa']-export_per_sector_2016['Vantagem Comparativa'].mean()) / export_per_sector_2016['Vantagem Comparativa'].std() < 3]
export_per_sector_2016 = export_per_sector_2016[(export_per_sector_2016['Vantagem Absoluta']-export_per_sector_2016['Vantagem Absoluta'].mean()) / export_per_sector_2016['Vantagem Absoluta'].std() < 3]


# In[21]:


# gráfico de vantagem comparativa vs vantagem absoluta com regra de regressão
sns.regplot(x='Vantagem Comparativa', y='Vantagem Absoluta', data=export_per_sector_2016)
plt.title('Regressão entre Vantagem Absoluta e Vantagem Comparativa')


# In[22]:


export_per_sector_2005_2015 = export_per_sector.query("year in [2005, 2015]")
export_per_sector_2005_2015


# In[23]:


export_per_sector_2005_2015['Log_RCA'] = np.log(export_per_sector_2005_2015['Vantagem Comparativa'])
export_per_sector_2005_2015


# In[24]:


export_per_sector_2005_2015 = export_per_sector_2005_2015.reset_index()
export_per_sector_2005_2015


# In[25]:


#################################################


# In[26]:


pivot_export_per_sector_2005_2015 = export_per_sector_2005_2015.pivot(index=['Country','Indústria'], columns='year', values=['Log_RCA','Vantagem Comparativa'])
pivot_export_per_sector_2005_2015


# In[27]:


pivot_export_per_sector_2005_2015.reset_index()
pivot_export_per_sector_2005_2015


# In[28]:


pivot_export_per_sector_2005_2015['Variação Log_RCA'] = pivot_export_per_sector_2005_2015.iloc[:, 1] - pivot_export_per_sector_2005_2015.iloc[:, 0]
pivot_export_per_sector_2005_2015


# In[29]:


##########################################################


# In[30]:


pivot_export_per_sector_2005_2015_1 = pivot_export_per_sector_2005_2015
pivot_export_per_sector_2005_2015_1


# In[31]:


pivot_export_per_sector_2005_2015_1.dropna(inplace=True)
pivot_export_per_sector_2005_2015_1


# In[32]:


export_per_sector_2005_2015.groupby(['Country', 'Indústria'])['Log_RCA'].diff()
export_per_sector_2005_2015


# In[33]:


new_df = pd.DataFrame(columns=['Diff_Log_RCA'])
new_df.loc[:, 'Diff_Log_RCA'] = export_per_sector_2005_2015.groupby(['Country', 'Indústria'])['Log_RCA'].diff().replace([np.inf, -np.inf], np.nan).dropna()
new_df = new_df.reset_index(drop=True)
new_df


# In[34]:


vantagem_comp_2005 = export_per_sector_2005_2015[export_per_sector_2005_2015['year']== 2005].groupby(['Country', 'Indústria'])['Vantagem Comparativa'].first().reset_index().replace([np.inf, -np.inf], np.nan)
vantagem_comp_2005_final_1 = vantagem_comp_2005.dropna()
vantagem_comp_2005_final_1


# In[38]:


vantagem_comp_2005_final_2 = vantagem_comp_2005_final_1.merge(new_df, left_index=True, right_index=True).set_index('Country')
vantagem_comp_2005_final_2 = vantagem_comp_2005_final_2.reset_index()
vantagem_comp_2005_final_2


# In[39]:


### Sem efeitos fixos

# Definindo as variáveis dependentes e independentes
X = vantagem_comp_2005_final_2['Diff_Log_RCA']
y = vantagem_comp_2005_final_2['Vantagem Comparativa']

# Adicionando uma constante
X = sm.add_constant(X)

# Fit no modelo de regressão
model = sm.OLS(y, X).fit()

#Print do resultado
print(model.summary())


# In[40]:


#Item h)

#### Efeito fixo por país

# Definindo as variáveis dependentes e independentes
X = vantagem_comp_2005_final_2['Diff_Log_RCA']
y = vantagem_comp_2005_final_2['Vantagem Comparativa']

# Adicionando uma constante e o efeito fixo de país exportador
X = pd.concat([sm.add_constant(X), pd.get_dummies(vantagem_comp_2005_final_2['Country'], drop_first=True)], axis=1)

# Fit no modelo de regressão com efeito fixo de país
model_c = sm.OLS(y, X, hasconst=True).fit()

#Print com o resultado
print(model_c.summary())


# In[42]:


#### Efeito fixo por indústria

# Definindo as variáveis dependentes e independentes
X = vantagem_comp_2005_final_2['Diff_Log_RCA']
y = vantagem_comp_2005_final_2['Vantagem Comparativa']

# Adicionando uma constante e o efeito fixo de país exportador
X = pd.concat([sm.add_constant(X), pd.get_dummies(vantagem_comp_2005_final_2['Indústria'], drop_first=True)], axis=1)

# Fit no modelo de regressão com efeito fixo de país
model_i = sm.OLS(y, X, hasconst=True).fit()

#Print com o resultado
print(model_i.summary())


# In[43]:


# Questão 2 


# In[44]:


# Tratando Base Do Pen Table


# In[47]:


pwt = pd.read_excel(r'com_int_dados_aps2/pwt1001.xlsx',sheet_name='Data')
pwt.head()


# In[48]:


# Pen Table Final


# In[49]:


pen_table_1 = pwt[(pwt['year']>=2000) & (pwt['year']<=2016)][['country','year','hc','cn','rnna']].set_index('country')
pen_table_1_axis_name = pen_table_1.rename_axis('Country')
pen_table_1_axis_name


# In[50]:


# Tratando Base do Banco Mundial


# In[53]:


wdi = pd.read_csv(r'com_int_dados_aps2/wdi.csv')
wdi.head()


# In[54]:


id_vars = ['countryname', 'countrycode', 'indicatorname', 'indicatorcode','region']

# Derreta as colunas de anos em sequência para a nova coluna "Year"
wdi_melted = pd.melt(wdi, id_vars=id_vars, var_name='year', value_name='Value')
wdi_melted['year'] = wdi_melted['year'].str.replace('v', '').astype(int)
wdi_melted.head()


# In[55]:


# BAse Final do Banco Mundial


# In[56]:


wdi_melted_final = wdi_melted[(wdi_melted['indicatorcode'] == 'AG.LND.AGRI.K2') & (wdi_melted['year']>=2000) & (wdi_melted['year']<=2016)][['countryname','indicatorcode','year','Value']].set_index('countryname')
wdi_melted_final_axis_name = wdi_melted_final.rename_axis('Country')
wdi_melted_final_axis_name


# In[57]:


# Tratando Base Anterior 


# In[58]:


export_per_sector.head()


# In[59]:


export_per_sector['Log_RCA'] = np.log(export_per_sector['Vantagem Comparativa'])
export_per_sector


# In[60]:


export_per_sector_reset_1 = export_per_sector.reset_index()
export_per_sector_reset_1


# In[61]:


# Base parte 2.1 Final 


# In[62]:


export_per_sector_reset_1_final = export_per_sector_reset_1[['Country','Indústria','year','Log_RCA']].replace([np.inf, -np.inf], np.nan).set_index('Country').dropna()
export_per_sector_reset_1_final


# In[63]:


# Juntando as Bases


# In[64]:


ultra_base_final_merged = pen_table_1_axis_name.merge(wdi_melted_final_axis_name, on=['Country', 'year'])                           .merge(export_per_sector_reset_1_final, on=['Country', 'year'])
ultra_base_final_merged_sem_na = ultra_base_final_merged.dropna()
ultra_base_final_merged_sem_na.head()


# In[65]:


# Mesma base, Porém agora para o ano de 2016


# In[66]:


ultra_base_final_merged_sem_na_2016 = ultra_base_final_merged_sem_na[ultra_base_final_merged_sem_na['year']== 2016]
ultra_base_final_merged_sem_na_2016 = ultra_base_final_merged_sem_na_2016.rename(columns={'Value': 'Tamanho da Terra Agricola'})
ultra_base_final_merged_sem_na_2016.head()


# In[67]:


# Realizando a Regressão


# In[68]:


# Definindo as variáveis dependentes e independentes
y = ultra_base_final_merged_sem_na_2016['Log_RCA']
X = ultra_base_final_merged_sem_na_2016['Tamanho da Terra Agricola']

# Adicionando uma constante
X = sm.add_constant(X)

# Fit no modelo de regressão
model_terras = sm.OLS(y, X).fit()

# Print 
print(model_terras.summary())


# In[69]:


# Removendo outliers
ultra_base_final_merged_sem_na_2016 = ultra_base_final_merged_sem_na_2016[(ultra_base_final_merged_sem_na_2016['Log_RCA']-ultra_base_final_merged_sem_na_2016['Log_RCA'].mean()) / ultra_base_final_merged_sem_na_2016['Log_RCA'].std() < 3]
ultra_base_final_merged_sem_na_2016 = ultra_base_final_merged_sem_na_2016[(ultra_base_final_merged_sem_na_2016['Tamanho da Terra Agricola']-ultra_base_final_merged_sem_na_2016['Tamanho da Terra Agricola'].mean()) / ultra_base_final_merged_sem_na_2016['Tamanho da Terra Agricola'].std() < 3]


# In[70]:


sns.regplot(x='Tamanho da Terra Agricola', y='Log_RCA', data=ultra_base_final_merged_sem_na_2016)
plt.title('Regressão entre Tamanho da Terra Agricola e Log_RCA')
plt.show()


# In[71]:


# Definindo as variáveis dependentes e independentes
y = ultra_base_final_merged_sem_na_2016['Log_RCA']
X = ultra_base_final_merged_sem_na_2016['hc']

# Adicionando uma constante
X = sm.add_constant(X)

# Fit no modelo de regressão
model_hc = sm.OLS(y, X).fit()

# Print 
print(model_hc.summary())


# In[72]:


sns.regplot(x='hc', y='Log_RCA', data=ultra_base_final_merged_sem_na_2016)
plt.title('Regressão entre Estoque de Capital Humano e Log_RCA')
plt.show()


# In[73]:


# Definindo as variáveis dependentes e independentes
y = ultra_base_final_merged_sem_na_2016['Log_RCA']
X = ultra_base_final_merged_sem_na_2016['cn']

# Definindo as variáveis dependentes e independentes
X = sm.add_constant(X)

# Fit no modelo de regressão
model_cn = sm.OLS(y, X).fit()

# Print 
print(model_cn.summary())


# In[74]:


sns.regplot(x='cn', y='Log_RCA', data=ultra_base_final_merged_sem_na_2016)
plt.title('Regressão entre Estoque de Capital e Log_RCA')
plt.show()

