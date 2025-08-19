import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Ruta a tu archivo .csv (ajusta si es necesario)
df = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\revised_cleaned.csv")

# Eliminar valores nulos en columnas clave
df = df.dropna(subset=["account", "product_type", "time_ref", "value", "status"])

# Separar en variables predictoras (X) y variable objetivo (y)
X = df[["account", "product_type", "time_ref", "value"]]
y = df["status"]

# Codificar variables categóricas y dejar 'value' tal cual
preprocesador = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["account", "product_type", "time_ref"])
], remainder="passthrough")

# Crear pipeline completo
modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("clasificador", RandomForestClassifier(random_state=42))
])

# Entrenar el modelo
modelo.fit(X, y)

# Guardar modelo completo (pipeline)
ruta_salida = r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\modelo_comercio.pkl"
with open(ruta_salida, "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo entrenado y guardado exitosamente.")