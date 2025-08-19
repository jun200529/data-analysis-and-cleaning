import pandas as pd
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import base64
import pydeck as pdk
import joblib

# Función para convertir imagen a base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Codificar imágenes como base64
talento_logo = get_img_as_base64("descarga.png")
udea_logo = get_img_as_base64("Escudo-UdeA.svg.png")
fondo_udea = get_img_as_base64("udea.jpg")

# Configuración de la página
st.set_page_config(layout="wide")

# Dividir en dos columnas: izquierda (azul), derecha (fondo UdeA)
col1, col2 = st.columns([1, 2])

# Columna izquierda
with col1:
    st.markdown(f"""
        <div style='
            background-color:#002857;
            height:100vh;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align: center;
            width: 100%;
        '>
            <img src='data:image/png;base64,{talento_logo}' width='200'><br><br>
            <h2 style='color:white; font-family:"Times New Roman"; text-align:center;'>
                Análisis de Datos<br>
                Nivel integrador
            </h2>
            <br>
            <a href='#inicio'>
                <button style='
                    font-size:18px;
                    padding:10px 30px;
                    background-color:white;
                    border:none;
                    border-radius:5px;
                    font-family:"Times New Roman";
                    font-weight:bold;
                    cursor:pointer;
                '>COMENCEMOS</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# Columna derecha con imagen de fondo y logo UdeA
with col2:
    st.markdown(f"""
        <div style='
            background-image: url("data:image/jpg;base64,{fondo_udea}");
            background-size: cover;
            background-position: center;
            height: 100vh;
            position: relative;
        '>
            <!-- Máscara blanca translúcida que cubre todo -->
            <div style='
                background-color: rgba(255, 255, 255, 0.7);
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                padding: 50px;
                box-sizing: border-box;
            '>
                <img src='data:image/png;base64,{udea_logo}' width='200'><br><br>
                <h1 style='font-family:"Times New Roman"; font-weight:bold; color: #002b5c;'>Comercio Internacional</h1>
                <p style='font-size:20px; font-family:"Times New Roman"; color: #002b5c;'>
                    Juan Sebastián Osorio Ortiz<br>
                    Manuela Galeano Chica
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Punto de inicio para el botón "COMENCEMOS"
st.markdown("<div id='inicio'></div>", unsafe_allow_html=True)

# Menú lateral
menu = st.sidebar.selectbox(
    "Navegación",
    ["Inicio", "Gráficas: país", "Gráficas: comercio","Machine learning", "Conclusiones"]
)

# Pagina 1: Portada
if menu == "Inicio":
    # Aquí va tu código de portada (el que hiciste con imágenes base64)
    st.markdown("<!-- Pega aquí tu portada -->", unsafe_allow_html=True)

# Pagina 2: graficas por paises
elif menu == "Gráficas: país":
    st.title("Análisis de Clasificación de Países")

    # Cargar los datos
    df_paises = pd.read_csv("country_classification_cleaned.csv")
    df_comercio = pd.read_csv("revised_cleaned.csv")

    
    #  Parte 1: Tabla de países
   
    st.subheader("Base de países clasificados")
    st.dataframe(df_paises)

    # Parte 2: Frecuencia real de países en el comercio

    st.subheader("Frecuencia de participación de países en el comercio")

    # Contar cuántas veces aparece cada país en los datos de comercio
    frecuencia = df_comercio['country_code'].value_counts().reset_index()
    frecuencia.columns = ['country_code', 'frecuencia']

    # Unir con nombres de países
    frecuencia = frecuencia.merge(df_paises, on='country_code', how='left')
    frecuencia = frecuencia.dropna(subset=['country_label'])
    frecuencia = frecuencia.sort_values(by='frecuencia', ascending=False)

    # Mostrar gráfico
    st.bar_chart(frecuencia.set_index('country_label')['frecuencia'])

    st.markdown("---")

    # Parte 3: Análisis de comercio por país (con picos reales)

    st.subheader("Análisis de comercio por país")

    # Selección de país
    pais_seleccionado = st.selectbox("Selecciona un país:", df_paises['country_label'].unique())
    codigo_pais = df_paises[df_paises['country_label'] == pais_seleccionado]['country_code'].values[0]

    # Filtrar comercio solo por ese país
    comercio_pais = df_comercio[df_comercio['country_code'] == codigo_pais]

    if comercio_pais.empty:
        st.warning(f"No hay datos de comercio para {pais_seleccionado}.")
    else:
        # Extraer años únicos
        años_disponibles = sorted(set(str(val)[:4] for val in comercio_pais['time_ref'].unique()))
        año_seleccionado = st.selectbox("Selecciona un año:", años_disponibles)

        # Filtrar registros cuyo time_ref comience por el año seleccionado
        comercio_anual = comercio_pais[comercio_pais['time_ref'].astype(str).str.startswith(año_seleccionado)]

        st.subheader(f"Registros de {pais_seleccionado} durante {año_seleccionado}")
        st.dataframe(comercio_anual)

        st.subheader(f"Serie de valores reales durante {año_seleccionado}")

        if len(comercio_anual) > 1:
            # Ordenar por time_ref y crear gráfico de línea con picos reales
            comercio_ordenado = comercio_anual.sort_values(by="time_ref").reset_index(drop=True)
            comercio_ordenado["Índice"] = comercio_ordenado.index  # para graficar como línea
            st.line_chart(comercio_ordenado.set_index("Índice")["value"])
        else:
            st.info("No hay suficientes registros para graficar evolución temporal.")

    st.markdown("---")

    # Parte 4: Top 10 países con mayor valor total acumulado

    st.subheader("países con mayor valor de exportacion y inportacion acumulado")

    resumen = df_comercio.groupby("country_code")["value"].sum().reset_index()
    resumen = resumen.merge(df_paises, on="country_code", how="left")
    top10_paises = resumen.sort_values(by="value", ascending=False).head(10)

    st.dataframe(top10_paises[["country_label", "value"]].rename(columns={
        "country_label": "País",
        "value": "Valor Total"
    }))

    
# Página 3: Gráficas de comercio
elif menu == "Gráficas: comercio":
    st.title("Comparación Exportaciones vs Importaciones")

    # --------------------
    # Sección 1: Preparación y uniones
    # --------------------
    df_comercio = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\revised_cleaned.csv")
    df_paises = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\country_classification_cleaned.csv")
    coord_df = pd.read_csv("world_coordinates.csv")

    df = df_comercio.merge(df_paises, on="country_code", how="left")

    # Cargar icono para el mapa
    def load_image_as_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    icon_base64 = load_image_as_base64("684908.png")
    icon_url = f"data:image/png;base64,{icon_base64}"

    # Agrupar para resumen global
    df_mapa = df.groupby(["country_label", "product_type", "account"])["value"].sum().reset_index()

    pivot = df_mapa.pivot_table(index="country_label", 
                                columns=["account", "product_type"], 
                                values="value", 
                                aggfunc="sum", 
                                fill_value=0)

    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    pivot["Exports"] = pivot.filter(like="Exports").sum(axis=1)
    pivot["Imports"] = pivot.filter(like="Imports").sum(axis=1)
    pivot["Goods"] = pivot.filter(like="Goods").sum(axis=1)
    pivot["Services"] = pivot.filter(like="Services").sum(axis=1)

    pivot = pivot.merge(coord_df, on="country_label", how="left").dropna(subset=["lat", "lon"])

    # --------------------
    # Sección 2: Análisis por país
    # --------------------
    pais_seleccionado = st.selectbox("Selecciona un país:", df["country_label"].dropna().unique())
    codigo_pais = df[df["country_label"] == pais_seleccionado]["country_code"].values[0]
    df_filtrado = df[df["country_code"] == codigo_pais]

    if df_filtrado.empty:
        st.warning("No hay datos disponibles para este país.")
    else:
        st.subheader(f"Distribución de Exportaciones vs Importaciones para {pais_seleccionado}")

        resumen = df_filtrado.groupby("account")["value"].sum().reset_index()
        st.dataframe(resumen.rename(columns={"account": "Tipo de Cuenta", "value": "Valor Total"}))

        total = resumen["value"].sum()
        resumen["porcentaje"] = (resumen["value"] / total * 100).round(1)

        idx_max = resumen["porcentaje"].idxmax()
        tipo_mayor = resumen.loc[idx_max, "account"]
        porcentaje_mayor = resumen.loc[idx_max, "porcentaje"]

        col1, col2 = st.columns([1, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            colores = ["#1E88E5", "#F9A825"]
            ax.pie(resumen["value"], labels=resumen["account"], autopct='%1.1f%%', startangle=90, colors=colores)
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            mensaje_detallado = (
                f"Las <b>{tipo_mayor}</b> representan el valor más alto del comercio de este país, "
                f"con un porcentaje de <b>{porcentaje_mayor}%</b> en relación al total entre exportaciones e importaciones."
            )

            st.markdown(
                f"""
                <div style='display: flex; height: 460px; justify-content: center; align-items: center; background-color: #002857; border-radius: 10px; padding: 30px;'>
                    <div style='font-size: 18px; font-family: "Segoe UI", sans-serif; text-align: center; line-height: 1.7; color: white;'>
                        <h2 style='margin-bottom: 20px; font-size: 28px; font-weight: bold;'>Resumen del Comercio Nacional</h2>
                        {mensaje_detallado}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Métricas clave del comercio para este país")

        m1, m2, m3 = st.columns(3)
        total_export = df_filtrado[df_filtrado["account"].str.lower().str.contains("export")].shape[0]
        total_import = df_filtrado[df_filtrado["account"].str.lower().str.contains("import")].shape[0]
        total_registros = df_filtrado.shape[0]
        diferencia = total_export - total_import

        m1.metric("Total Exportaciones (registros)", f"{total_export:,}", delta=f"{diferencia:+,} frente a import.")
        m2.metric("Total Importaciones (registros)", f"{total_import:,}", delta=f"{-diferencia:+,} frente a export.")
        m3.metric("Registros totales", f"{total_registros:,}", delta=f"{df_filtrado['time_ref'].nunique()} trimestres registrados")

        # --------------------
        # Sección 3: Productos y mapa
        # --------------------
        st.subheader("Análisis Detallado: Productos y Mapa Global")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### exportaciones/importaciones")
            df_top = df_filtrado.dropna(subset=["product_type", "value"])
            top_productos = df_top.groupby(["account", "product_type"])["value"].sum().reset_index()

            top_exports = top_productos[top_productos["account"].str.lower().str.contains("export")]
            top_imports = top_productos[top_productos["account"].str.lower().str.contains("import")]

            st.write("**exportaciones**")
            st.dataframe(top_exports.sort_values(by="value", ascending=False).head(5), use_container_width=True)

            st.write("**importaciones**")
            st.dataframe(top_imports.sort_values(by="value", ascending=False).head(5), use_container_width=True)

        with col2:
            st.markdown("#### Mapa Interactivo del Comercio Global")

            df_mapa_simple = pivot[["country_label", "Exports", "Imports", "lat", "lon"]].copy()
            import pydeck as pdk

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_mapa_simple,
                get_position='[lon, lat]',
                get_radius=200000,
                get_fill_color=[0, 100, 255, 180],
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(latitude=10, longitude=0, zoom=1.2)

            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{country_label}</b><br/>\nExportaciones: <b>{Exports:,.0f}</b><br/>\nImportaciones: <b>{Imports:,.0f}</b>",
                    "style": {"color": "white", "backgroundColor": "#002857", "padding": "8px"}
                }
            ))

# Página 4: Machine Learning
elif menu == "Machine learning":
    st.title("Predicción de Finalización de Comercio")

    # --- Cargar datos base para inputs y país ---
    df_comercio = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\revised_cleaned.csv")
    df_paises = pd.read_csv(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\country_classification_cleaned.csv")
    df = df_comercio.merge(df_paises, on="country_code", how="left")

    # --- Entradas para el usuario ---
    st.subheader("Inserta los datos para predecir si el registro estará Finalizado (F) o en Curso (C):")
    st.markdown("""
    <div style='font-size:18px; line-height:1.8; text-align: justify;'>
    Este modelo de Machine Learning ha sido diseñado para predecir si un registro comercial, es decir, una operación de comercio internacional realizada por un país, estará en estado <b>Finalizado (F)</b> o <b>en Curso (C)</b> en función de múltiples características relevantes. Entre los atributos considerados se encuentran el trimestre de operación (<i>time_ref</i>), el tipo de cuenta involucrada (exportación o importación), el código del producto, el país que realiza la operación, el tipo de producto y el valor económico asociado. 
    <br><br>
    Esto permite que, al ingresar nuevos datos —por ejemplo, predicciones para trimestres futuros como el segundo trimestre de 2025 o 2026— el sistema pueda estimar con una alta probabilidad si esas transacciones estarán completas o no, basándose en el comportamiento del comercio real en años anteriores. 
    <br><br>
    """, unsafe_allow_html=True)


    # Opciones disponibles
    años = [f"{a}-Q{q}" for a in [2025, 2026] for q in range(1, 5)]
    cuentas = df["account"].dropna().unique()
    tipos_producto = df["product_type"].dropna().unique()
    codigos_disponibles = df["country_code"].dropna().unique()

    # Inputs del usuario
    time_ref = st.selectbox("Trimestre futuro (time_ref):", años)
    account = st.selectbox("Tipo de cuenta:", sorted(cuentas))
    code = st.text_input("Código del producto (ej. 123456):", "100000")
    country_code = st.selectbox("Código del país:", sorted(codigos_disponibles))
    product_type = st.selectbox("Tipo de producto:", sorted(tipos_producto))
    value = st.number_input("Valor (en millones):", min_value=0.0, step=0.1)

    # Botón para predecir
    if st.button("Predecir"):
        try:
            # --- Cargar modelo entrenado ---
            import pickle
            with open(r"C:\Users\youca\OneDrive\Desktop\SQL\Proyecto final\modelo_comercio.pkl", "rb") as archivo:
                modelo = pickle.load(archivo)

            # --- Crear DataFrame del usuario con las columnas correctas ---
            columnas_modelo = ["time_ref", "account", "code", "country_code", "product_type", "value"]
            df_usuario = pd.DataFrame([{
                "time_ref": time_ref,
                "account": account,
                "code": code,
                "country_code": country_code,
                "product_type": product_type,
                "value": value
            }], columns=columnas_modelo)

            # --- Predicción ---
            prediccion = modelo.predict(df_usuario)[0]
            proba = modelo.predict_proba(df_usuario)[0]

            st.markdown(f"### Resultado de la Predicción:")
            st.markdown(f"- **Estado estimado**: `{prediccion}`")
            st.markdown(f"- **Probabilidad Finalizado (F):** {proba[1]*100:.2f}%")
            st.markdown(f"- **Probabilidad en Curso (C):** {proba[0]*100:.2f}%")

            # --- Mostrar gráficas adicionales por país ---
            st.markdown("### Comportamiento histórico del país seleccionado")
            df_pais = df[df["country_code"] == country_code]

            if df_pais.empty:
                st.warning("No hay registros históricos para este país.")
            else:
                col1, col2, col3 = st.columns(3)


                with col1:
                    st.markdown("**Distribución por tipo de cuenta**")
                    cuentas_data = df_pais["account"].value_counts()
                    st.bar_chart(cuentas_data)

                with col2:
                    st.markdown("**Distribución por tipo de producto**")
                    tipos_data = df_pais["product_type"].value_counts()
                    st.bar_chart(tipos_data)

                with col3:
                    st.markdown("**Total de valores por trimestre**")
                    resumen_tiempo = df_pais.groupby("time_ref")["value"].sum().sort_index()
                    st.line_chart(resumen_tiempo)

        except Exception as e:
            st.error(f"⚠️ Error al predecir: {e}")

    
# Página 5: Conclusiones
elif menu == "Conclusiones":
    st.markdown("""
        <h1 style='text-align: center; color: #002857;'>Conclusiones del Análisis</h1>
        <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: justify; font-size: 18px; line-height: 1.8;'>
    Durante el primer trimestre de 2025, el comercio internacional de <b>Nueva Zelanda</b> estuvo ligeramente inclinado hacia las <b>importaciones</b>, que representaron el <b>53.63%</b> del total comercial, frente a un <b>46.37%</b> de exportaciones. Eso muestra un leve desequilibrio en la balanza comercial, con más compras que ventas al exterior. Se registraron más de <b>1.43 millones</b> de operaciones de importación y <b>1.07 millones</b> de exportación, lo que refleja un alto volumen de transacciones.
    <br><br>
    Tanto en importaciones como en exportaciones, los <b>bienes físicos</b> dominaron el intercambio, con una participación del <b>74.7%</b> en importaciones y <b>70.2%</b> en exportaciones. Eso indica que la economía sigue moviéndose principalmente con mercancías como alimentos, maquinaria o materias primas. Sin embargo, los <b>servicios</b> también tienen peso, sobre todo en exportaciones, donde alcanzan casi el <b>30%</b>, lo cual muestra una economía parcialmente diversificada hacia sectores como turismo, educación o tecnología que también aportan al país.
    <br><br>
    <b>En cuanto a los países socios comerciales más importantes:</b>
    <ul>
        <li><b>Importaciones:</b> el principal proveedor fue <b>Australia (AU)</b>, con el <b>7.28%</b> del valor total importado, seguido muy de cerca por <b>China (CN)</b> con el <b>7.06%</b>.</li>
        <li><b>Exportaciones:</b> el mayor destino también fue <b>China (CN)</b>, consolidándose como el principal socio comercial del país, seguido por <b>Australia (AU)</b> y <b>Estados Unidos (US)</b>.</li>
    </ul>
    Esto nos da una visión de que <b>Nueva Zelanda</b> mantiene relaciones comerciales fuertes con sus vecinos regionales, y aunque sigue dependiendo de bienes físicos, también se abre paso en la <b>exportación de servicios</b>.
    </div>
    """, unsafe_allow_html=True)

