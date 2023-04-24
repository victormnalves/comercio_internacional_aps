#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import * 
import geopandas as gpd


# In[40]:


##################################################2.1######################################################
#Item a)
#Importação da base de dados wdi:
wdi = pd.read_csv(r'com_int_dados_aps1/wdi.csv')
wdi


# In[3]:


#Selecionando as colunas relevantes:
wdi_a = wdi[['countryname', 'countrycode', 'indicatorname', 'v2016']]
wdi_a


# In[4]:


#Filtragem dos dados para obtenção de apenas a população e o PIB em 2016
wdi_a = wdi_a[(wdi_a['indicatorname'] == 'Population, total') | (wdi_a['indicatorname'] == 'GDP (current US$)')]
wdi_a

#Renomeando v2016:
wdi_a = wdi_a.rename(columns={'v2016':'2016'})
wdi_a


# In[5]:


#Pivotando a tabela para ter cada país em uma linha e as informações em colunas:
wdi_a = wdi_a.pivot(index='countryname', columns='indicatorname', values='2016').reset_index()
wdi_a


# In[6]:


#Renomeando as colunas:
wdi_a.columns = ['Country', 'GDP', 'Population']
wdi_a

#Conversão do PIB para bilhões de U$:
wdi_a['GDP'] = wdi_a['GDP'] / 10**9
wdi_a


# In[7]:


##############################################Item b)##########################################################


# In[8]:


##Importação da base de dados itpd:
itpd = pd.read_stata(r'com_int_dados_aps1/itpd.dta')
itpd


# In[9]:


#Exclusão do comércio doméstico:
itpd = itpd[itpd['exporter_iso3'] != itpd['importer_iso3']]
itpd


# In[10]:


#Selecionando apenas o ano de 2016:
itpd_2016 = itpd.loc[itpd['year'] == 2016]
itpd_2016


# In[11]:


#Criando uma base para exportações:
itpd_e = itpd_2016[['exporter_m49','industry_id','broad_sector','trade']]
itpd_e.rename(columns = {'exporter_m49':'Country'}, inplace = True)
itpd_e


# In[12]:


#Criando uma base para importações:
itpd_i = itpd_2016[['importer_m49','industry_id','broad_sector','trade']]
itpd_i.rename(columns = {'importer_m49':'Country'}, inplace = True)
itpd_i


# In[13]:


# selecionando variáveis relevantes
itpd_e = itpd_2016[['exporter_m49', 'broad_sector', 'trade']] 
# renomeando as variáveis
itpd_e = itpd_e.rename(columns={'exporter_m49': 'Country', 'trade': 'exports'})
# modificando os tipos das variáveis
itpd_e['Country'] = itpd_e['Country'].astype('category')
itpd_e['broad_sector'] = itpd['broad_sector'].astype('category')
# convertendo para bilhões de U$:
itpd_e['exports'] = pd.to_numeric(itpd_e['exports'])/10**3
# ordenando os dados por variáveis categóricas para facilitar junção das bases
itpd_e = itpd_e.sort_values(by=['Country', 'broad_sector']).reset_index(drop=True)

# selecionando variáveis relevantes
itpd_i = itpd_2016[['importer_m49', 'broad_sector', 'trade']]
# renomeando as variáveis
itpd_i = itpd_i.rename(columns={'importer_m49': 'Country2', 'trade': 'imports', 'broad_sector': 'broad_sector2'})
# modificando os tipos das variáveis
itpd_i['Country2'] = itpd_i['Country2'].astype('category')
itpd_i['broad_sector2'] = itpd_i['broad_sector2'].astype('category')
#Convertendo para bilhões de U$:
itpd_i['imports'] = pd.to_numeric(itpd_i['imports'])/10**3
# ordenando os dados por variáveis categóricas para facilitar junção das bases
itpd_i = itpd_i.sort_values(by=['Country2', 'broad_sector2']).reset_index(drop=True)
# unindo bases de exportação e importação
itpd_ie = pd.concat([itpd_e, itpd_i], axis=1)
itpd_ie.head(10)


# In[14]:


# selecionando apenas variáveis relevantes
itpd_ie = itpd_ie[['Country', 'broad_sector', 'exports', 'imports']]
# criando a balança comercial
itpd_ie['trade_balance'] = itpd_ie['exports'] - itpd_ie['imports']
itpd_ie.head(10)


# In[15]:


# Criando importações e exportações totais por país e por setor

#Agregando variáveis numéricas:
itpd_ie_sector = itpd_ie.groupby(['Country', 'broad_sector']).agg(exports=('exports', 'sum'), imports=('imports', 'sum'), trade_balance=('trade_balance', 'sum')).reset_index()
# renomeando as colunas
itpd_ie_sector.columns = ['Country', 'broad_sector', 'exports', 'imports', 'trade_balance']
# transformando a bases em um dataframe
itpd_ie_sector = pd.DataFrame(itpd_ie_sector)
# calculando o comércio total
itpd_ie_sector['total_commerce'] = itpd_ie_sector['exports'] + itpd_ie_sector['imports']

itpd_ie_sector.head(10)


# In[16]:


#Criando importações e exportações totais por país
itpd_ie_total = itpd_ie.groupby(['Country']).agg(exports=('exports', 'sum'), imports=('imports', 'sum'), trade_balance=('trade_balance', 'sum')).reset_index()
# renomeando as colunas
itpd_ie_total.columns = ['Country', 'exports', 'imports', 'trade_balance']
# transformando a bases em um dataframe
itpd_ie_total = pd.DataFrame(itpd_ie_total)
# calculando o comércio total
itpd_ie_total['total_commerce'] = itpd_ie_total['exports'] + itpd_ie_total['imports']

itpd_ie_total.head(10)


# In[17]:


#Combinando informações de comércio por setor e país e do Banco Mundial:
final_database_sector = pd.merge(wdi_a, itpd_ie_sector, on = 'Country', how = 'right')
final_database_sector.head()


# In[18]:


#Combinando informações de comércio por país e do Banco Mundial:
final_database_total = pd.merge(wdi_a, itpd_ie_total, on = 'Country', how = 'right')
final_database_total


# In[19]:


#Criando uma nova coluna com os dados em ln para exportações e importações
final_database_total["ln(exports)"] = np.log(final_database_total["exports"])
final_database_total["ln(imports)"] = np.log(final_database_total["imports"])
final_database_total


# In[20]:


#####################################2.2####################################################################


# In[21]:


#########################################Item a)############################################################
ggplot(final_database_total, aes(x="ln(exports)", y="ln(imports)", size="GDP")) +     geom_point(alpha=0.8, fill="darkblue") +     labs(x="Ln Exportações (bilhões de U$)", y="Ln Importações (bilhões de U$)",          title="Exportações x Importações ponderada pelo PIB") +     geom_smooth(method="lm", color="red", se=False, show_legend=False) +     theme_classic()


# In[22]:


##########################################Item b)###########################################################
final_database_total["total_commerce/GDP"] = final_database_total["total_commerce"]/final_database_total["GDP"]
final_database_total

final_database_total["ln(GDP)"] = np.log(final_database_total["GDP"])
final_database_total


# In[23]:


#Gráfico com outliers:
(ggplot(final_database_total, aes(x = "ln(GDP)", y = "total_commerce/GDP")) +      geom_point(alpha = 0.8, color = "darkblue") +      labs(x = "Log do PIB (bilhões de U$)", y = "Comércio total/PIB (bilhões de U$)",
         title = "Comércio total como relação do PIB x log PIB") + \
     geom_smooth(method="lm", color="red", se=False, show_legend=False) + \
     theme_classic())


# In[24]:


#Gráfico sem outliers:
(ggplot(final_database_total, aes(x = "ln(GDP)", y = "total_commerce/GDP")) +      geom_point(alpha = 0.8, color = "darkblue") +      labs(x = "Log do PIB (bilhões de U$)", y = "Comércio total/PIB (bilhões de U$)",
         title = "Comércio total como relação do PIB x log PIB") + \
     geom_smooth(method="lm", color="red", se=False, show_legend=False) + \
     ylim(0,20) + \
     theme_classic())


# In[25]:


##########################################2.3#############################################################
#Item a)
#Retirando os dados desnecessários ao que pede o item:
final_database_sector = final_database_sector.drop(columns=["trade_balance", "total_commerce"])
final_database_sector["exports/pop"] = final_database_sector["exports"]/final_database_sector["Population"]
final_database_sector["imports/pop"] = final_database_sector["imports"]/final_database_sector["Population"]
final_database_sector.head(10)


# In[26]:


final_a_exports_pc= final_database_sector.pivot(index='Country', columns='broad_sector', values='exports/pop').reset_index()
final_a_exports_pc.head(50)


# In[27]:


final_a_imports_pc= final_database_sector.pivot(index='Country', columns='broad_sector', values='imports/pop').reset_index()


# In[28]:


#Item b)
import geopandas as gpd

#leitura dos dados geográficos de cada país
map_data = gpd.read_file(r'com_int_dados_aps1\ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
map_data

#Renomeando "SOVEREIGNT" para "Country"
map_data = map_data.rename(columns = {"SOVEREIGNT" : "Country"})


# In[29]:


#Base agregada com os dados geográficos e de exportações per capita
final_a_exports_pc_map = pd.merge(map_data, final_a_exports_pc, on='Country')
final_a_exports_pc_map.head(10)

#Base agregada com os dados geográficos e de importações per capita
final_a_imports_pc_map = pd.merge(map_data, final_a_imports_pc, on='Country')
final_a_imports_pc_map.head(10)


# In[31]:


from mpl_toolkits.axes_grid1 import make_axes_locatable 

#Removendo a Antártica, pois influencia negativamente na plotagem do mapa.
world= final_a_imports_pc_map[(final_a_imports_pc_map.Services>0) & (final_a_imports_pc_map.Country!="Antarctica") & (final_a_imports_pc_map['Mining & Energy']>0) & (final_a_imports_pc_map['Manufacturing']>0) & (final_a_imports_pc_map['Agriculture']>0)]


# In[74]:


import mapclassify
from mpl_toolkits.axes_grid1 import make_axes_locatable

q10 = mapclassify.Quantiles(world.Services, k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world.assign(cl=q10.yb).plot(
    column="Services",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()

#Adicionando legenda ao mapa mundi.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = []  #Criando um array vazio para evitar erro de vazio
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Services", fontsize=14)

# Adicionando título
plt.title("Importações Per Capita pelo Setor de Serviços", fontsize=13)

# Mostrar o plot
plt.show()


# In[71]:


q10 = mapclassify.Quantiles(world.Agriculture, k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world.assign(cl=q10.yb).plot(
    column="Agriculture",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = []  
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Agriculture", fontsize=14)


plt.title("Importações Per Capita pelo Setor de Agricultura", fontsize=13)


plt.show()


# In[72]:


q10 = mapclassify.Quantiles(world['Mining & Energy'], k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world.assign(cl=q10.yb).plot(
    column="Mining & Energy",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = []  
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Mining & Energy", fontsize=14)


plt.title("Importações Per Capita pelo Setor de Energia e Mineração", fontsize=13)


plt.show()


# In[73]:


q10 = mapclassify.Quantiles(world['Manufacturing'], k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world.assign(cl=q10.yb).plot(
    column="Manufacturing",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = []  
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Manufacturing", fontsize=14)


plt.title("Importações Per Capita pelo Setor de Manufatura", fontsize=13)


plt.show()


# In[36]:


world_ex= final_a_exports_pc_map[(final_a_exports_pc_map.Services>0) & (final_a_exports_pc_map.Country!="Antarctica") & (final_a_exports_pc_map['Mining & Energy']>0) & (final_a_exports_pc_map['Manufacturing']>0) & (final_a_exports_pc_map['Agriculture']>0)]


# In[76]:


q10 = mapclassify.Quantiles(world_ex['Mining & Energy'], k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world_ex.assign(cl=q10.yb).plot(
    column="Mining & Energy",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = [] 
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Mining & Energy", fontsize=14)


plt.title("Exportações Per Capita pelo Setor de Mineração & Energia", fontsize=13)


plt.show()


# In[68]:


q10 = mapclassify.Quantiles(world_ex.Manufacturing, k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world_ex.assign(cl=q10.yb).plot(
    column="Manufacturing",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = [] 
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Manufacturing", fontsize=14)


plt.title("Exportações Per Capita pelo Setor de Manufatura", fontsize=13)


plt.show()


# In[70]:


q10 = mapclassify.Quantiles(world_ex.Agriculture, k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world_ex.assign(cl=q10.yb).plot(
    column="Agriculture",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = [] 
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Agriculture", fontsize=14)

plt.title("Exportações Per Capita pelo Setor de  Agricultura", fontsize=13)

# Show the plot
plt.show()


# In[75]:


q10 = mapclassify.Quantiles(world_ex.Agriculture, k=5)

fig, ax = plt.subplots(figsize=(16, 9))
world_ex.assign(cl=q10.yb).plot(
    column="Agriculture",
    categorical=True,
    k=2,
    cmap="OrRd",
    linewidth=1,
    edgecolor="white",
    ax=ax
)
ax.set_axis_off()


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=q10.yb.min(), vmax=q10.yb.max()))
sm._A = [] 
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("Services", fontsize=14)

plt.title("Exportações Per Capita pelo Setor de Serviço", fontsize=13)

# Show the plot
plt.show()


# In[63]:


#Item c

# Agrupar os dados por setor
final_database_sector_filtrada = final_database_sector.drop(['GDP', 'Population','imports','exports/pop','imports/pop'], axis=1)
setores = final_database_sector_filtrada.groupby('broad_sector')

# Classificar os grupos em ordem decrescente com base nos valores de exportação absolutos
setores_export = setores.apply(lambda x: x.nlargest(5, ['exports']))
setores_export


# In[65]:


# Agrupar os dados por setor
final_database_sector_filtrada = final_database_sector.drop(['GDP', 'Population','exports','exports/pop','imports/pop'], axis=1)
setores = final_database_sector_filtrada.groupby('broad_sector')

# Classificar os grupos em ordem decrescente com base nos valores de importações absolutos
setores_export = setores.apply(lambda x: x.nlargest(5, ['imports']))
setores_export


# In[66]:


# Agrupar os dados por setor
final_database_sector_filtrada = final_database_sector.drop(['GDP', 'Population','exports', 'imports','imports/pop'], axis=1)
setores = final_database_sector_filtrada.groupby('broad_sector')

# Classificar os grupos em ordem decrescente com base nos valores de exportações per capita
setores_export = setores.apply(lambda x: x.nlargest(5, ['exports/pop']))
setores_export


# In[67]:


# Agrupar os dados por setor
final_database_sector_filtrada = final_database_sector.drop(['GDP', 'Population','exports', 'exports/pop','imports'], axis=1)
setores = final_database_sector_filtrada.groupby('broad_sector')

# Classificar os grupos em ordem decrescente com base nos valores de importações per capita
setores_export = setores.apply(lambda x: x.nlargest(5, ['imports/pop']))
setores_export


# In[78]:


itpd_brazil = itpd_ie_sector[itpd_ie_sector['Country'] == 'Brazil']
itpd_brazil
total_export = itpd_brazil['exports'].sum()
total_export
itpd_brazil['% Sector in Total Exports'] = (itpd_brazil['exports']/total_export) * 100
itpd_brazil
# Configurando o tamanho da figura
plt.figure(figsize=(8, 8))

# Criando o gráfico de setores
plt.pie(itpd_brazil['% Sector in Total Exports'], labels=itpd_brazil['broad_sector'], autopct='%1.1f%%')

# Título
plt.title('% Sector in Total Exports for Brazil')

# Mostrando o gráfico
plt.show()
total_import = itpd_brazil['imports'].sum()
itpd_brazil['% Sector in Total Imports'] = (itpd_brazil['imports']/total_export) * 100


plt.figure(figsize=(8, 8))


plt.pie(itpd_brazil['% Sector in Total Imports'], labels=itpd_brazil['broad_sector'], autopct='%1.1f%%')


plt.title('% Sector in Total Imports for Brazil')


plt.show()
itpd_india = itpd_ie_sector[itpd_ie_sector['Country'] == 'India']
itpd_india
total_export_india = itpd_india['exports'].sum()
total_export_india
itpd_india['% Sector in Total Exports'] = (itpd_india['exports']/total_export_india) * 100
itpd_india

plt.figure(figsize=(8, 8))


plt.pie(itpd_india['% Sector in Total Exports'], labels=itpd_india['broad_sector'], autopct='%1.1f%%')


plt.title('% Sector in Total Exports for India')


plt.show()
total_imports_india = itpd_india['imports'].sum()
itpd_india['% Sector in Total Imports'] = (itpd_india['imports']/total_export) * 100


plt.figure(figsize=(8, 8))


plt.pie(itpd_india['% Sector in Total Imports'], labels=itpd_india['broad_sector'], autopct='%1.1f%%')


plt.title('% Sector in Total Imports for India')


plt.show()

