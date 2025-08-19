import pandas as pd

# Cargar el archivo original
revised_df = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\revised.csv")

# 1. Eliminar duplicados
revised_cleaned = revised_df.drop_duplicates()

# 2. Eliminar filas con valores nulos
revised_cleaned = revised_cleaned.dropna()

# 3. Asegurar que la columna 'value' sea numérica
revised_cleaned['value'] = pd.to_numeric(revised_cleaned['value'], errors='coerce')

# 4. Convertir 'time_ref' a formato de fecha (asumiendo el día 01 del mes)
revised_cleaned['date'] = pd.to_datetime(revised_cleaned['time_ref'].astype(str) + "01", format="%Y%m%d")

# 5. Guardar archivo limpio
revised_cleaned.to_csv("revised_cleaned.csv", index=False)