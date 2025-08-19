import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#analizar datos
ruta='revised.csv'
data = pd.read_csv(ruta)
print(data.shape)
data.head()
data.info() 

#ahora vamos a limpiarlos datos
#datos faltantes
data.dropna(inplace=True)
data.info()

#Columnas irrelevantes
#conteo de los niveles en las diferentes columnas categoricas
cols_cat= ["time_ref","account","code","country_code","product_type","value","status"]
for col in cols_cat:
    print(f'columna {col}: {data[col].nunique()} subniveles')
    #como tinen mas de 1 nivel, lo mejor es no eliminarlas

data.describe()

#filas repetidas (no hay)
print(f'Tamaño del set antes de eliminar duplicados: {data.shape}')
data.drop_duplicates(inplace=True)
print(f'Tamaño del set despues de eliminar duplicados: {data.shape}')

#outliers - valores extremos
#generar graficas individuales para cada columna numerica
cols_num=["time_ref","account","code","country_code","product_type","value","status"]

fig, ax= plt.subplots(nrows=7, ncols=1, figsize=(8,30))
fig.subplots_adjust(hspace=0.5)

for i, col in enumerate(cols_num):
    sns.boxplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)

# colab
"""
observacion de las graficas
- time_ref: apaarecen muy grandes de 202420, 202480, etc
- account: valores entre 0 y 1000, no hay outliers (todo esta literalmente junto)
- code: valores entre 0 y 1000 (esta bien)
- country_code: valores entre 0 y 1000 (esta bien)
- product_type: supongo que bien
- value: hay valores negativos, no deberian (corregir)
- status: supongo que bien
"""
#eliminar filas con valores negativos en la columna value
print(f'Tamaño del set antes de eliminar outliers: {data.shape}')
data = data[data['value'] >= 0]
print(f'Tamaño del set despues de eliminar outliers: {data.shape}')

#errores tipograficos en variables categoricas
#Unknown /=/ UNK
cols_cat= ["time_ref","account","code","country_code","product_type","value","status"]
fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(10,30))
fig.subplots_adjust(hspace=1)

for i, col in enumerate(cols_cat):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)

for column in data.columns:
    if column in cols_cat:
        data[column] = data[column].str.lower()

fig, ax= plt.subplots(nrows=10, ncols=1, figsize=(10,30))
fig.subplots_adjust(hspace=1)

for i, col in enumerate(cols_cat):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)

#aquie el unifica job: amdin. y administrative
#primer forma
print(data[].unique())
data[''] = data[''].str.replace('admin.', 'administrative', regex =True)
print (data[''].unique())

#matiral: unificar div. divorced 
#segunda forma
print(data['education'].unique())
data['']=data[''].str.replace
data[data['']=='unk'] = 'unknown'
print (data[''].unique())




