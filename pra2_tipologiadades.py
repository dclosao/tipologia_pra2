#!/usr/bin/env python
# coding: utf-8

# # Pràctica 2
# Presentació:
# Hem seleccionat el joc de dades del Titànic donat que compta amb un gran nombre de variables numèriques i categòriques i el tamany total del joc de dades és bastant bo per a realitzar tasques de processament de dades i d'anàlisi exploratori de dades.
# 
# # Índex
# 1. [Descripció del joc de dades](#part1)
# 2. [Integració i selecció de les dades d’interès a analitzar](#part2)
# 3. [Neteja de dades](#part3)
#     1. [Valors perduts](#part3_2)
#     2. [Valors extrems](#part3_3)
# 4. [Anàlisi de dades](#part4)
#     1. [Visualització de dades](#part4_2)
#     2. [Anàlisi exploratori dels supervivents](#part4_3)
# 5. [Anàlisi Estadístic](#part5)
#     1. [Nomralitat](#part5_2)
#     2. [Homogeneïtat de la variància](#part5_3)
#     3. [Correlació entre variables](#part5_4)
# 6. [Conclusions finals de l'estudi](#part6)
# 7. [Contribució als resultats](#part7)
# 
# 

# ## Descripció del joc de dades <a name="#part1"></a>
# 
# * Descripció curta: el joc de dades conté informació sobre el tipus de passatger del Titànic, les característiques del seu viatge i si va sobreviure o no.
# * URL de descàrrega: https://data.world/nrippner/titanic-disaster-dataset
# * Mida del joc de dades: 1309 files, 14 columnes
# * Columnes:
#     * survival: binari, 0=NO, 1=Sí. Booleà
#     * class: tipus de passatger, 1=de primera, 2=de segona, 3=de tercera. Enter
#     * name: nom del passatger. Cadena de text
#     * sex: sexe del passatger. Cadena de text
#     * age: edat en anys. Decimal
#     * sibsp: número d'esposes o germans. Enter
#     * parch: número de pares o fills. Enter
#     * ticket: número del tíquet. Enter
#     * fare: preu del billet. Decimal
#     * cabin: cabina. Cadena de text
#     * embarked: port d'embarcament (C = Cherbourg, Q = Queenstown, S = Southampton). Cadena de text
#     * boat: bot salva-vides en el cas d'haver sobreviscut. Cadena de text
#     * body: número del cos en el cas de no haver sobreviscut i que s'hagi trobat el cos. Enter
#     * home_dest: destinació del passatger. Cadena de text
# 
# **Quina pregunta pretén respondre?**
# 
# La pregunta principal que tractarem de respondre en aquest projecte és si existeixen relacions entre les característiques del passatgers i el seu índex de supervivència.
# 
# **Objectiu:**
# Per tant, com acabem de comentar l'objectiu principal és conèixer si existeixen relacions entre el tipus de passatger i el seu índex de supervivència.
# 
# 

# ## Integració i selecció de les dades d’interès a analitzar <a name="#part2"></a>
# Importem les llibreries necessàries per a treballar en aquest projecte:

# In[1]:


# importació de llibreries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import *
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import kstest
import statsmodels.api as sm
import pylab
import random
from matplotlib import pyplot


# Llegim el joc de dades i mostrem els les primeres files:

# In[2]:


df = pd.read_csv('titanic.csv')
df.head(3)


# In[3]:


df.shape


# ## Neteja de dades<a name="#part3"></a> 
# 
# Fem un cop d'ull als valors perduts del joc de dades:

# ### Valors perduts<a name="#part3_2"></a> 

# In[4]:


df.isnull().sum()


# In[5]:


#Com que només hi ha 1 valor missing de fare el podem intentar substituir per la mitjana
df.describe()[['fare']]


# In[6]:


df[['pclass']][df['fare'].isna()]
#La pclass del missing es 3


# In[7]:


#El preu esta molt influenciat per la classe, aleshores anem a veure separat per aixó
df[['fare','pclass']].groupby('pclass').mean()


# In[8]:


#Aleshores substituim el valor missing de fare per 13.3


# **Com tractarem els valors perduts?**
# * age: donat que és rellevant per a aquest estudi procedirem a eliminar les files que no contiguin l'edat.
# * fare: substituim per 13.3
# * cabin: no és rellevant. Omplirem els valors perduts amb "desconegut".
# * embarked: és rellevant amb el que també s'eliminaran les files que continguin aquests valors.
# * boat, body i home_dest: no és rellevant. Omplirem els valors perduts amb "desconegut".
# 
# #### Tractament dels valors perduts:
# #### Eliminació de files
# Eliminem les files que continguin valors perduts a age o a embarked. Primer comprovem quin tant per cent d'informació perdíem a l'eliminar-los, si és poc procedirem a eliminar:

# In[9]:


# fem el recompte de files
total_files = df.shape[0]

# fem el recompte de valors perduts a les dues columnes
nan_age = df.age.isna().sum()
nan_embarked = df.embarked.isna().sum()
nan_fare = df.fare.isna().sum()

# veiem quin tant per cent d'informació perdrem
print("La columna age té {} valors perduts d'un total de {}, eliminant aquestes files eliminarem el {}% de les dades."
      .format(nan_age, total_files, nan_age / total_files * 100))
print("La columna embarked té {} valors perduts d'un total de {}, eliminant aquestes files eliminarem el {}% de les dades."
      .format(nan_embarked, total_files, nan_embarked / total_files * 100))
print("La columna fare té {} valors perduts d'un total de {}, eliminant aquestes files eliminarem el {}% de les dades."
      .format(nan_fare, total_files, nan_fare / total_files * 100))


# Tenint en compte aquesta pèrdua d'informació procedim a eliminar els valors perduts.

# In[10]:


# eliminem files amb valors perduts
df = df.dropna(subset=['age', 'embarked'])

total_files = df.shape[0]
print("Total de files: ", total_files)


# In[11]:


df[['fare']] = df[['fare']].fillna(value=13.3)


# #### Substitució dels valors perduts
# En el cas de cabin, boat, body i home_dest procedirem a substituir, com hem comentat, els valors perduts per la paraula "desconegut".

# In[12]:


df[['cabin', 'boat', 'body', 'home_dest']] = df[['cabin', 'boat', 'body', 'home_dest']].fillna(value='desconegut')


# Veiem com ha quedat el sumatori dels valors perduts al joc de dades:

# In[13]:


df.isnull().sum()


# In[ ]:





# ### Valors Extrems<a name="#part3_3"></a> 
# 

# **Tenim 3 possibles tractaments pels valors extrems:**
# 
#     -Eliminar les files corresponents a aquets valors
#     -Definir limits superiors i/o inferiors i substituir els valors extrems pels valors limits
#     -Tractar-los com valors missings (i substituir per la mitjana)

# **Fem una primera vizualització dels valors extrems de les variables numeriques**

# In[14]:


fig = plt.figure(num=None, figsize=(12, 6), facecolor='w')
#Afegim també els boxplot de cada variable
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)
ax1.title.set_text('age')
ax1.boxplot(df['age'])
ax2.title.set_text('fare')
ax2.boxplot(df['fare'])
ax3.title.set_text('parch')
ax3.boxplot(df['parch'])
ax4.title.set_text('sibsp')
ax4.boxplot(df['sibsp'])
plt.show()


# Observem que les variables Age i Fare poden tenir valors extrems

# In[15]:


df.describe()[['fare', 'age']]


# Estudiem en funció de pclass

# In[16]:


df.groupby('pclass').describe()[['fare']]


# Analitzarem visualment la variable fare per pclass

# In[17]:


# gràficament també podem observar això

mr1=df['fare'][df['pclass']==1]
mr2=df['fare'][df['pclass']==2]
mr3=df['fare'][df['pclass']==3]

plt.subplots(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.hist(list(mr3), bins=20, label='Class 3',linewidth=1, edgecolor='black',fc=(1, 0, 0, 0.5))
plt.title('Fare in Class 3')

plt.subplot(1, 3, 2)
plt.hist(list(mr2), bins=20, label='Class 2',linewidth=1, edgecolor='black',fc=(1, 1, 0, 0.5))
plt.title('Fare in Class 2')

plt.subplot(1, 3, 3)
plt.hist(list(mr1), bins=20, label='Class 1',linewidth=1, edgecolor='black',fc=(0, 0, 1, 0.5))
plt.title('Fare in Class 1')

plt.tight_layout(2)
plt.show()


# **Amb l'histograma observem els següents outliers:**
# 
#     -Cas 1: en la pclass 1 tenim un outlier en el 500
#     -Cas 2: em la pclass 2 tenim dos outliers amb fare>60
#     -Cas 3: em la pclass 3 tenim tres outliers amb fare>40
#     
# Per l'altra banda observem que tant la pclass 1 i 3 tenen fare casos amb fare igual a 0. Considerem que aquests valors són erronis i els substituirem per la mitjana.
# 
# 
# 
# 

# In[18]:


df.groupby('pclass').describe()[['age']]


# In[19]:


# gràfica

ar1=df['age'][df['pclass']==1]
ar2=df['age'][df['pclass']==2]
ar3=df['age'][df['pclass']==3]

plt.subplots(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.hist(list(ar1), bins=20, label='Class 3',linewidth=1, edgecolor='black',fc=(1, 0, 0, 0.5))
plt.title('Age in Class 3')

plt.subplot(1, 3, 2)
plt.hist(list(ar2), bins=20, label='Class 2',linewidth=1, edgecolor='black',fc=(1, 1, 0, 0.5))
plt.title('Age in Class 2')

plt.subplot(1, 3, 3)
plt.hist(list(ar3), bins=20, label='Class 1',linewidth=1, edgecolor='black',fc=(0, 0, 1, 0.5))
plt.title('Age in Class 1')

plt.tight_layout(2)
plt.show()


# In[20]:


#Separat per sexe tampoc veiem diferencies

get_ipython().run_line_magic('matplotlib', 'inline')
saF=df['age'][df['sex']=='female']
saM=df['age'][df['sex']=='male']
fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8,6))
pyplot.hist(list(saF), bins=20, edgecolor='black', linewidth=1, label='female', fc=(0, 0, 1, 0.5))
pyplot.hist(list(saM), bins=20, edgecolor='black', linewidth=1, label='male',fc=(1, 0, 0, 0.5))
pyplot.legend(loc='upper right')
pyplot.title('Age')
pyplot.show()


# Considerem que per l'edad no hi han valors extrems

# #### Tractament dels valors extrems:
# 

# ### Cas 1

# In[21]:


df.loc[df.loc[:, 'fare'] >500]


# In[22]:


df = df[df.fare < 500]

#Eliminem les files amb els valors extrems del cas 1.


# In[23]:


df.shape


# ### Cas 2

# In[24]:


df=df[((df.fare <60)  & (df.pclass==2)) | (df.pclass!=2)]


# In[25]:


df.shape


# ### Cas 3

# In[26]:


df=df[((df.fare <50)  & (df.pclass==3)) | (df.pclass!=3)]


# In[27]:


df.shape


# ### Tractaments del fare = 0

# Primer fem una primera vista dels casos amb fare=0.

# In[28]:


df.loc[(df['fare'] == 0),:]


# In[29]:


cond1 = (df['fare'] == 0) & (df['pclass'] == 1) 
cond3 = (df['fare'] == 0) & (df['pclass'] == 3) 


# In[30]:


df.loc[cond1,'fare'] = 87.50
df.loc[cond3,'fare'] = 13.30


# In[31]:


df.groupby('pclass').describe()[['fare']]


# ### Expulsió del fitxer

# In[32]:


### creem un csv amb el joc de dades tractat
df.to_csv('output_titanic.csv', encoding='utf-8-sig', index=False)


# # Anàlisi de dades<a name="#part4"></a> 
# 
# ## Visualització de dades<a name="#part4_2"></a> 
# Veiem primer les distribucions de les columnes principals per a tenir una idea més acurada del joc de dades.

# In[33]:


# columna classe
print('Passatgers per tipus de classe:')
df['pclass'].value_counts().plot(kind='pie',autopct='%.2f')


# In[34]:


# columna age i sex
print("Quina és la distribució per edats i sexe?")

as_fig = sns.FacetGrid(df,hue='sex',aspect=5)

as_fig.map(sns.kdeplot,'age',shade=True)

oldest = df['age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# In[35]:


print("Recompte de persones en funció del sexe i de la classe: ")
sns.catplot(x='pclass',
            data=df, 
            hue='sex', 
            kind='count')


# In[36]:


print("Des d'on van embarcar i quin tipus de classe de passatgers s'hi van pujar?")
sns.catplot(x='embarked',data=df,
               hue='pclass',
               kind='count')


# ## Anàlisi exploratori dels supervivents<a name="#part4_3"></a> 

# In[37]:


# creem una funció que ens compari dues columnes i ens mostri un gràfic de barres

def graficar_dues_columnes(df, 
                            col1, 
                            col2, 
                            legloc='upper right',
                            plt_style = 'gggraficar',
                            color_palette="dark", 
                            sorter=None, 
                            stacked=False,
                            kind = 'bar', 
                            percentage = True,
                            custom_title=None, 
                            minimal=True, 
                            figsize=(14,6), width=0.6):   
    
    # creació de la funció del gràfic
    def graficar(table, 
            legloc='upper right',
            plt_style = 'seaborn-ticks',
            color_palette="dark",
            sorter=None, 
            stacked=False,
            kind = 'bar', 
            percentage = True,
            custom_title=None, 
            minimal=True, 
            figsize=(19,10), 
            width=0.7 ):     
        agrupat = table

        if percentage == True:
            agrupat = np.round(agrupat.divide(agrupat['Total'],axis=0)*100,0)
        try:   
            del agrupat['Total']
        except:
            pass

        if sorter:
            agrupat = agrupat[sorter]

        plt.style.use(plt_style)
        sns.set_palette(sns.color_palette(color_palette))
        ax = agrupat.plot(kind=kind,stacked=stacked, figsize=figsize, width=width)
        _ = plt.setp(ax.get_xticklabels(), rotation=0)
        plt.legend(loc=legloc)

        if percentage == True:
          for p in ax.patches:
                ax.annotate('{}%'.format(int(np.round(p.get_height(),decimals=2))),
                                             (p.get_x()+p.get_width()/2.,
                                              p.get_height()), ha='center', va='center',
                                            xytext=(0, 10), textcoords='offset points')
        else:
          for p in ax.patches:
                ax.annotate(np.round(p.get_height(),decimals=2),
                                             (p.get_x()+p.get_width()/2.,
                                              p.get_height()), ha='center', va='center',
                                            xytext=(0, 10), textcoords='offset points')
        if minimal == True:
            ax.get_yaxis().set_ticks([])
            plt.xlabel('')
            sns.despine(top=True, right=True, left=True, bottom=False);
        else:
            pass     
        plt.title(custom_title)
    
    
    
    # creació del gràfic
    agrupat = df.groupby([col2,col1]).size().unstack(col2)
    
    agrupat['Total'] = agrupat.sum(axis=1)
   
    graficar(agrupat, 
            legloc=legloc,
            plt_style = plt_style,
            color_palette=color_palette,
            sorter=sorter, 
            stacked=stacked,
            kind = kind, 
            percentage = percentage,
            custom_title=custom_title, 
            minimal=minimal, 
            figsize=figsize, 
            width=width)  


# In[38]:


# columna survived
print('Quin tant per cent va sobreviure?')
df['survived'].value_counts().plot(kind='pie', autopct='%.2f')


# In[39]:


# sex i survived
graficar_dues_columnes(df,
                       'sex', 
                       'survived', 
                       color_palette=('bisque','darkorange'),
                       plt_style = 'seaborn-ticks', 
                       custom_title='Tant per cent de supervivents en funció del sexe')


# In[40]:


# embarked i survived
graficar_dues_columnes(df,
                       'embarked', 
                       'survived', 
                       color_palette=('lightcyan','dodgerblue'),
                       plt_style = 'seaborn-ticks', 
                       custom_title='Tant per cent de supervivents en funció del port on van embarcar')


# In[41]:


# pclass i survived
graficar_dues_columnes(df,
                       'pclass', 
                       'survived', 
                       color_palette=('mistyrose','salmon'),
                       plt_style = 'seaborn-ticks', 
                       custom_title='Tant per cent de supervivents en funció de la classe')


# In[42]:


# columna age i sex
print("Supervivents per edat")

as_fig = sns.FacetGrid(df,hue='survived',aspect=5)

as_fig.map(sns.kdeplot,'age',shade=True)

oldest = df['age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# ## Anàlisi Estadístic<a name="#part5"></a> 

# ### Nomralitat <a name="#part5_2"></a> 

# #### - Variable Age

# #### Test de Kolmogorov-Smirnov per la normalitat

# In[43]:


from scipy.stats import kstest

#perform Kolmogorov-Smirnov test
kstest(df['age'], 'norm')


# #### Test de e Shapiro-Wilk per la normalitat

# In[44]:



shapiro_test = stats.shapiro(df['age'])
shapiro_test


# #### Veiem que pvalue es inferior a 0.05 tan en el test de KS con en el test de S-W. Aleshores rebutjem la hipòtesi nul·la i es conclou que l'edad no té distribució normal.

# #### - Variable fare

# In[45]:



#perform Kolmogorov-Smirnov test
kstest(df['fare'], 'norm')


# In[46]:


from scipy import stats
stats.shapiro(df['fare'])


# #### Veiem que pvalue es inferior a 0.05 tan en el test de KS con en el test de S-W. Aleshores rebutjem la hipòtesi nul·la i es conclou que el preu no té distribució normal.

# In[47]:


# normalitzadió per desviació típica
norm_df=(df-df.mean(numeric_only=True))/df.std(numeric_only=True)


#  ### Homogeneïtat de la variància<a name="#part5_3"></a> 

# #### Comprobarem la homogeneïtat de la variància de les variables age i fare pels diferents grups de les variables pclass, sex i survived:

# #### -Variable age:

# In[48]:


#Estudiarem les variables age i fare en els grups: pclass, sex i surived
#Com que no son variables normals hem d'aplicar el test no parametric de Fligner-Killeen:

age_class1=df['age'][df['pclass']==1]
age_class2=df['age'][df['pclass']==2]
age_class3=df['age'][df['pclass']==3]
age_sexF=df['age'][df['sex']=='female']
age_sexM=df['age'][df['sex']=='male']
age_survived0=df['age'][df['survived']==0]
age_survived1=df['age'][df['survived']==1]
fligner_age_class = stats.fligner(age_class1, age_class2,age_class3, center='median')
fligner_age_sex = stats.fligner( age_sexF,age_sexM, center='median')
fligner_age_survived = stats.fligner(age_survived0, age_survived1, center='median')


# In[49]:


print ("El valor de p-value del test de age en la variable pclass es:", fligner_age_class.pvalue) 


# In[50]:


print ("El valor de p-value del test de age en la variable sex es:", fligner_age_sex.pvalue) 


# In[51]:


print ("El valor de p-value del test de age en la variable survived es:", fligner_age_survived.pvalue) 


# #### Observem que menys en el grup pclass la variable age no rebujta la hipotesis nul-la i per tant te variàncies estadísticament similars. 

# #### -Variable Fare:

# In[52]:


#Estudiarem les variables fare i fare en els grups: pclass, sex i surived
#Com que no son variables normals hem d'aplicar el test no parametric de Fligner-Killeen:
from scipy import stats
fare_class1=df['fare'][df['pclass']==1]
fare_class2=df['fare'][df['pclass']==2]
fare_class3=df['fare'][df['pclass']==3]
fare_sexF=df['fare'][df['sex']=='female']
fare_sexM=df['age'][df['sex']=='male']
fare_survived0=df['fare'][df['survived']==0]
fare_survived1=df['fare'][df['survived']==1]
fligner_fare_class = stats.fligner(fare_class1, fare_class2,fare_class3, center='median')
fligner_fare_sex = stats.fligner( fare_sexF,fare_sexM, center='median')
fligner_fare_survived = stats.fligner(fare_survived0, fare_survived1, center='median')


# In[53]:


print ("El valor de p-value del test de fare en la variable pclass es:", fligner_fare_class.pvalue) 


# In[54]:


print ("El valor de p-value del test de fare en la variable sex es:", fligner_fare_sex.pvalue) 


# In[55]:


print ("El valor de p-value del test de fare en la variable survived es:", fligner_fare_survived.pvalue) 


# ##### D'altre banda, per la variable fare en els grups estudiats el p-value es inferior a 0.05 i per tant es rebutja la hipòtesi nul·la d’homoscedasticitat i es conclou que la variable fare presenta variàncies estadísticament diferents per als diferents grups de pclass, sex i survived.

# ### Correlació entre variables<a name="#part5_4"></a> 

# In[56]:


# apliquem la fòrmula de correlació al joc de dades normalitzat
corr = norm_df.corr()

# creem el mapa de calor
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, 
            square=True, 
            mask=mask,
            linewidth=2.5, 
            vmax=0.4, vmin=-0.4, 
            cmap=cmap, 
            cbar=False, 
            ax=ax,
            annot=True)

ax.set_yticklabels(ax.get_xticklabels(), fontfamily='serif', rotation = 0, fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)

ax.spines['top'].set_visible(True)

plt.tight_layout()
plt.title("Mapa de calor de les correlacions entre variables")
plt.show()


# La correlació més alta la trobem entre classe i preu del billet, mentres que la més baixa la trobem entre el número de persones o germans i la variable survived.

# # Conclusions finals de l'estudi <a name="#part6"></a> 
# 
# L'objectiu principal d'aquest estudi tal i com hem comentat a l'inici era conèixer si existeixen relacions entre les diferents característiques de cada passatger i el tant per cent de supervivència de cada grup.
# En base a l'anàlisi visual en el que hem vist el tant per cent de supervivents de les principals variables categòriques podem extraure que:
# * El percentatge de mortaldat dels homes va ser molt superior al de les dones, tenin aquests un 80% i les dones tan sols un 25%.
# * L'únic port d'entrada que va tenir més supervivents que morts va ser C. Aquesta relació tampoc implica causalitat.
# * Hem pogut veure com, en funció de la classe en la que viatjaven, quan més alta era més tant per cent de supervivents hi havien. 
# 
# En base a l'anàlisi de les variables numèriques hem vist que:
# * La única correlació forta existeix entre la classe i la supervivència. És quelcom que hem pogut veure a l'anàlisi visual i amb l'anàlisi de correlació s'ha reforçat.
# * La segona correlació més alta és entre el preu del tíquet i la supervivència.
# * Les altres variables numèriques tenen correlacions dèbils o gairebé inexistents amb la supervivència.

# # Contribució als resultats<a name="#part7"></a> 
# 
# 

# | Contibucions  | Firma  |
# |---|---|
# |Investigació previa   |    cdediu & dclosao   |
# |Integració de dades   |  dclosao             |
# | Neteja de dades  |      dclosao      |
# | Tractament dels valors extrems  | cdediu  |
# | Anàlisi de dades  | cdediu & dclosao   |
# | Anàlisi de normalitat  | dclosao  |
# | Anàlisi de Homogeneïtat  | cdediu  |
# | Correlació entre variables  | cdediu & dclosao     |
# | Conclusións finals  | cdediu & dclosao     |
# 

# In[ ]:




