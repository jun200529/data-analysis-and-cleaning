import pandas as pd

# Cargar el archivo original
country_classification_df = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\country_classification.csv")

# 1. Eliminar duplicados
country_classification_cleaned = country_classification_df.drop_duplicates()

# 2. Eliminar filas con valores nulos
country_classification_cleaned = country_classification_cleaned.dropna()

country_classification_df

# 3. Guardar archivo limpio
country_classification_cleaned.to_csv("country_classification_cleaned.csv", index=False)

