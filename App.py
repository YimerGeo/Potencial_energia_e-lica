import logging
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import geocoder
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Montar la carpeta estática
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ruta al archivo CSV
path = "DataSet/Velocidad_Viento_Antioquia_Magdalena.csv"

# Cargar y limpiar los datos
def load_wind_data():
    try:
        logger.info("Cargando datos desde el archivo CSV...")
        df = pd.read_csv(path, on_bad_lines="skip")
        logger.info("Datos cargados correctamente.")
        
        df['municipality'] = df['Municipio'].str.lower()
        df['ValorObservado'] = pd.to_numeric(df['ValorObservado'], errors='coerce')
        df = df.dropna(subset=['ValorObservado', 'FechaObservacion', 'CodigoEstacion', 'Latitud', 'Longitud'])
        df = df.rename(columns={'ValorObservado': 'observed_value', 
                                'FechaObservacion': 'observation_date', 
                                'CodigoEstacion': 'station_code'})
        
        df['observation_date'] = pd.to_datetime(df['observation_date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        # Clasificar las velocidades del viento
        df['classification'] = df['observed_value'].apply(classify_wind_speed)
        # Manejo de NaN antes de entrenar modelos
        df['observed_value'] = df['observed_value'].fillna(df['observed_value'].mean())  # Imputación por la media
        df['station_code'] = df['station_code'].fillna(df['station_code'].mode()[0])  # Imputación por la moda para 'station_code'
        
        
        logger.info("Datos limpiados y preparados.")
        return df
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

# Asegúrate de crear la carpeta si no existe
os.makedirs('static/graficas', exist_ok=True)
# Establecer estilo visual de seaborn
# Configuración de estilo de Seaborn
# Configuración del estilo de Seaborn
sns.set(style="whitegrid", palette="pastel")


def haversine(lat1, lon1, lat2, lon2):
    # Radio de la Tierra en kilómetros
    R = 6371.0
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Fórmula de Haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distancia en kilómetros
    print(f"Distancia calculada entre ({lat1}, {lon1}) y ({lat2}, {lon2}): {distance:.2f} km")
    return distance

def calculate_similarity(df, distance_threshold=10):
    print(f"Calculando similitud entre estaciones con un umbral de distancia de {distance_threshold} km.")
    # Crear un DataFrame para almacenar las probabilidades
    prob_df = []
    
    for i, station1 in df.iterrows():
        print(f"Comparando estación {station1['station_code']}.")
        similar_stations = df[df['station_code'] != station1['station_code']]  # Excluir la misma estación
        for j, station2 in similar_stations.iterrows():
            # Calcular la distancia entre las estaciones
            distance = haversine(station1['Latitud'], station1['Longitud'], 
                                 station2['Latitud'], station2['Longitud'])
            
            if distance <= distance_threshold:  # Solo considerar estaciones cercanas
                # Evaluar la similitud de las velocidades del viento
                observed_value_diff = abs(station1['observed_value'] - station2['observed_value'])
                prob = np.exp(-observed_value_diff)  # Usar una función exponencial para la probabilidad
                print(f"Similitud entre estación {station1['station_code']} y {station2['station_code']}: Probabilidad = {prob:.4f}")
                
                prob_df.append({
                    'station_code_1': station1['station_code'],
                    'station_code_2': station2['station_code'],
                    'distance': distance,
                    'observed_value_1': station1['observed_value'],
                    'observed_value_2': station2['observed_value'],
                    'probability': prob
                })
                
    print("Cálculo de similitudes completado.")
    return pd.DataFrame(prob_df)

# Función para calcular la distancia entre cada par de estaciones
def calculate_distances(df):
    print("Calculando las distancias entre todas las estaciones.")
    distances = []
    for i, station1 in df.iterrows():
        print(f"Calculando distancias para la estación {station1['station_code']}.")
        for j, station2 in df.iterrows():
            if station1['station_code'] != station2['station_code']:  # Evitar comparar la misma estación consigo misma
                distance = haversine(station1['Latitud'], station1['Longitud'], 
                                     station2['Latitud'], station2['Longitud'])
                distances.append({
                    'station_code_1': station1['station_code'],
                    'station_code_2': station2['station_code'],
                    'distance': distance
                })
    print("Cálculo de distancias completado.")
    return pd.DataFrame(distances)


# Entrenar modelo de Naive Bayes
def train_naive_bayes_model(df):
    try:
        mapping = {'Mala': 0, 'Regular': 1, 'Buena': 2}
        X = df[['observed_value']].values
        y = df['classification'].map(mapping).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logger.info(f"Precisión del modelo: {accuracy:.2f}")

        # Graficar y guardar la imagen
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Naive Bayes'], y=[accuracy], color=sns.color_palette()[0], width=0.5)  # Ajusta el ancho aquí
        plt.ylim(0, 1)
        plt.ylabel('Precisión', fontsize=14)
        plt.title('Precisión del Modelo Naive Bayes', fontsize=16)

        os.makedirs('static/graficas', exist_ok=True)

        plt.savefig('static/graficas/naive_bayes.png', bbox_inches='tight')
        plt.close()

        return model, accuracy  
    except Exception as e:
        logger.error(f"Error al entrenar el modelo: {e}")
        return None, None  

# Entrenar modelo de predicción temporal (por meses)
def train_regression_model(df):
    try:
        df['month'] = df['observation_date'].dt.month
        X = df[['month', 'station_code']].values
        y = df['observed_value'].values

        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        y_imputed = SimpleImputer(strategy='mean').fit_transform(y.reshape(-1, 1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

        model_regression = LinearRegression()
        model_regression.fit(X_train, y_train)

        model_hgb = HistGradientBoostingRegressor()
        model_hgb.fit(X_train, y_train)

        regression_score = model_regression.score(X_test, y_test)
        hgb_score = model_hgb.score(X_test, y_test)

        # Graficar y guardar la imagen
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Regresión Lineal', 'HistGradientBoosting'], 
                    y=[regression_score, hgb_score], 
                    palette=["#FFA07A", "#20B2AA"])
        
        plt.ylim(0, 1)
        plt.ylabel('Precisión', fontsize=14)
        plt.title('Precisión de Modelos de Regresión', fontsize=16)
        
        os.makedirs('static/graficas', exist_ok=True)
        
        plt.savefig('static/graficas/regresion.png', bbox_inches='tight')
        plt.close()

        return model_regression, model_hgb, X_test, y_test, regression_score, hgb_score  
    except Exception as e:
        logger.error(f"Error al entrenar los modelos: {e}")
        return None, None, None, None, None, None

def plot_and_save_graph(naive_bayes_accuracy, regression_score, hgb_score):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Graficar Naive Bayes con barra más delgada
        sns.barplot(ax=ax[0], x=['Naive Bayes'], y=[naive_bayes_accuracy], 
                    color=sns.color_palette()[0], width=0.5)  # Ajusta el ancho aquí
        ax[0].set_ylim(0, 1)
        ax[0].set_ylabel('Precisión', fontsize=14)
        ax[0].set_title('Precisión del Modelo Naive Bayes', fontsize=16)

        # Graficar Regresión con barra más delgada
        sns.barplot(ax=ax[1], x=['Regresión Lineal', 'HistGradientBoosting'], 
                    y=[regression_score, hgb_score], 
                    palette=["#FFA07A", "#20B2AA"], width=0.5)  # Ajusta el ancho aquí
        
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel('Precisión', fontsize=14)
        ax[1].set_title('Precisión de Modelos de Regresión', fontsize=16)

        plt.tight_layout()  
        plt.savefig('static/graficas/modelos_combinados.png', bbox_inches='tight')  
        plt.close()
    except Exception as e:
        logger.error(f"Error al graficar: {e}")

# Función para clasificar velocidades del viento
def classify_wind_speed(value):
    if value < 3.0:
        return 'Mala'
    elif 3.0 <= value <= 4.0:
        return 'Regular'
    else:
        return 'Buena'


# Cargar los datos al inicio de la aplicación para evitar cargarlo en cada solicitud.
wind_data_df = load_wind_data()
# Reemplaza con el código real de la estación

# Entrenar modelo de Naive Bayes
naive_bayes_model, accuracy_naive_bayes = train_naive_bayes_model(wind_data_df)

# Entrenar modelo de regresión
regression_model, model_hgb, X_test, y_test, regression_score, hgb_score = train_regression_model(wind_data_df)

# Graficar resultados combinados
plot_and_save_graph(accuracy_naive_bayes, regression_score, hgb_score)

# Variables globales para almacenar las coordenadas del usuario y el municipio
coordenadas_usuario = None
municipio_usuario = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    global coordenadas_usuario
    global municipio_usuario
    global nombre_usuario

    # Reiniciar las variables globales
    coordenadas_usuario = None
    municipio_usuario = None
    nombre_usuario = None

    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ver_csv", response_class=HTMLResponse)
async def ver_csv(request: Request):
    # Cargar datos del CSV
    try:
        # Cargar solo las columnas necesarias (ajusta según tus necesidades)
        columns_to_load = ['FechaObservacion', 'CodigoEstacion', 'ValorObservado', 'Municipio',]  # Ejemplo de columnas
        df = pd.read_csv(path, usecols=columns_to_load)  # Cargar solo columnas específicas

        # Convertir a diccionario solo si es necesario
        data = df.to_dict(orient='records')  # Convertir a diccionario
        return templates.TemplateResponse("ver_csv.html", {"request": request, "data": data})
    except Exception as e:
        logger.error(f"Error al leer el archivo CSV: {e}")
        raise HTTPException(status_code=500, detail="Error al cargar los datos del CSV.")

@app.get("/chatbot", response_class=HTMLResponse)
async def chat_bot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# Endpoint para acceder a las gráficas
@app.get("/graficas/modelos_combinados", response_class=FileResponse)
def obtener_grafica_combinada():
    ruta_grafica = "static/graficas/modelos_combinados.png"
    if os.path.exists(ruta_grafica):
        return FileResponse(ruta_grafica)
    else:
        raise HTTPException(status_code=404, detail="Gráfica no encontrada")

@app.get("/ubicacion", tags=["Ubicación"])
def obtener_ubicacion():
    global coordenadas_usuario
    try:
        coordenadas_usuario = obtener_ubicacion_usuario()
        logger.info(f"Coordenadas del usuario obtenidas: {coordenadas_usuario}")
        return {
            "Latitud": coordenadas_usuario[0],
            "Longitud": coordenadas_usuario[1],
        }
    except Exception as e:
        logger.error(f"Error al obtener la ubicación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def obtener_ubicacion_usuario():
    g = geocoder.ip('me')  # Obtiene la ubicación del cliente mediante la IP
    if g.latlng:
        lat, lon = g.latlng
        return lat, lon
    else:
        raise HTTPException(status_code=404, detail="No se pudo obtener la ubicación.")

@app.get("/estaciones/{municipio}", tags=["Estaciones"])
def obtener_estaciones_por_municipio(municipio: str):
    try:
        logger.info(f"Buscando estaciones para el municipio: {municipio}")
        
        estaciones_municipio = wind_data_df[wind_data_df['municipality'] == municipio.lower()]

        if estaciones_municipio.empty:
            raise HTTPException(status_code=404, detail="No se encontraron estaciones para el municipio especificado.")

        estaciones_unicas = estaciones_municipio[['station_code', 'Latitud', 'Longitud']].drop_duplicates()

        resultados = []
        
        for _, row in estaciones_unicas.iterrows():
            resultados.append({
                "station_code": row['station_code'],
                "latitude": row['Latitud'],
                "longitude": row['Longitud']
            })

        logger.info(f"Estaciones únicas encontradas: {len(resultados)}")
        
        return {
            "municipio": municipio,
            "estaciones": resultados
        }
    except Exception as e:
        logger.error(f"Error en la búsqueda: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/municipios/clasificacion/{municipality}", tags=["Clasificacion"])
def clasificar_municipio(municipality: str):
    if wind_data_df.empty:
        raise HTTPException(status_code=500, detail="Los datos no están disponibles.")

    municipality_data = wind_data_df[wind_data_df['municipality'].str.lower() == municipality.lower()]
    if municipality_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el municipio: {municipality}")

    avg_speed = municipality_data['observed_value'].mean()
    prediction = naive_bayes_model.predict([[avg_speed]])[0]
    classification = {0: 'Mala', 1: 'Regular', 2: 'Buena'}[prediction]

    station_count = len(municipality_data)
    max_value = municipality_data['observed_value'].max()
    min_value = municipality_data['observed_value'].min()

    return {
        "municipality": municipality,
        "station_count": station_count,
        "max_value": max_value,
        "min_value": min_value,
        "average_speed": avg_speed,
        "classification": classification
    }

@app.get("/estaciones/cercanas/{municipio}", tags=["Estaciones Cercanas"])
def obtener_estaciones_cercanas(municipio: str):
    try:
        if coordenadas_usuario is None:
            raise HTTPException(status_code=400, detail="Primero debe obtener la ubicación del usuario.")

        logger.info(f"Coordenadas del usuario: {coordenadas_usuario}")

        estaciones_info = obtener_estaciones_por_municipio(municipio)

        if not estaciones_info["estaciones"]:
            raise HTTPException(status_code=404, detail="No se encontraron estaciones para el municipio especificado.")

        radio_km = 5  # Distancia para considerar estaciones cercanas
        estaciones_cercanas = []
        estaciones_fuera_radio = [] 

        for estacion in estaciones_info["estaciones"]:
            coords_estacion = (estacion["latitude"], estacion["longitude"])
            distancia = geodesic(coordenadas_usuario, coords_estacion).kilometers

            estacion["distancia"] = distancia

            if distancia <= radio_km:
                estaciones_cercanas.append(estacion)
            else:
                estaciones_fuera_radio.append(estacion)

            logger.info(f"Estación: {estacion['station_code']} - Distancia: {distancia:.2f} km")

        # Ordenar todas las estaciones por distancia
        todas_estaciones = sorted(
            estaciones_cercanas + estaciones_fuera_radio,
            key=lambda e: e["distancia"]
        )

        logger.info(f"Total de estaciones cercanas encontradas: {len(estaciones_cercanas)}")
        logger.info(f"Total de estaciones encontradas fuera del radio: {len(estaciones_fuera_radio)}")

        return {
            "municipio": municipio,
            "estaciones_cercanas": estaciones_cercanas,
            "todas_estaciones": todas_estaciones,  # Incluye todas las estaciones ordenadas por distancia
            "total_cercanas": len(estaciones_cercanas),
            "total_todas": len(todas_estaciones)
        }

    except Exception as e:
        logger.error(f"Error al obtener estaciones cercanas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    
@app.get("/probabilidad_viento/{municipio}", tags=["Probabilidad Viento"])
async def obtener_probabilidad_viento(municipio: str):
    try:
        # Obtener las coordenadas del usuario (asumimos que están definidas)
        if coordenadas_usuario is None:
            raise HTTPException(status_code=400, detail="Primero debe obtener la ubicación del usuario.")
        
        logger.info(f"Coordenadas del usuario: {coordenadas_usuario}")

        # Obtener las estaciones cercanas del municipio
        estaciones_info = obtener_estaciones_cercanas(municipio)
        
        # Si no se encontraron estaciones dentro del radio de 5 km, tomamos todas las estaciones encontradas
        if len(estaciones_info["estaciones_cercanas"]) == 0:
            logger.info(f"No se encontraron estaciones cercanas a menos de 5 km. Tomando todas las estaciones encontradas.")
            estaciones_info["estaciones_cercanas"] = estaciones_info["todas_estaciones"]
        
        # Cargar los datos de viento
        df = load_wind_data()  # Cargar los datos desde el CSV (y limpiarlos)

        probabilidad_resultados = []

        # Recorrer cada estación cercana
        for estacion in estaciones_info["estaciones_cercanas"]:
            station_code = estacion["station_code"]
            distancia = estacion["distancia"]

            # Filtrar los datos de la estación actual
            datos_estacion = df[df['station_code'] == station_code]
            
            if datos_estacion.empty:
                logger.warning(f"No se encontraron datos de viento para la estación {station_code}")
                continue

            # Obtener el valor observado del viento de esa estación
            observed_value = datos_estacion['observed_value'].values[0]

            # Calcular la probabilidad de repetición según el modelo
            probabilidad = np.exp(-distancia) * observed_value  # Simplificación del modelo probabilístico

            probabilidad_resultados.append({
                "station_code": station_code,
                "distancia": distancia,
                "observed_value": observed_value,
                "probabilidad": probabilidad
            })

        # Retornar las probabilidades calculadas
        return {
            "municipio": municipio,
            "probabilidades_viento": probabilidad_resultados,
            "total_estaciones": len(probabilidad_resultados)
        }

    except Exception as e:
        logger.error(f"Error al obtener probabilidad de viento: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# En el endpoint de predicción, usar el modelo con mejor rendimiento
@app.get("/prediccion/{station_code}", tags=["Prediccion"])
def predecir_velocidad_viento(station_code: int):
    if wind_data_df.empty:
        raise HTTPException(status_code=500, detail="Los datos no están disponibles.")

    station_data = wind_data_df[wind_data_df['station_code'] == station_code]
    if station_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el código de estación: {station_code}")

    # Evaluar cuál modelo tiene mejor precisión
    if regression_score > hgb_score:
        best_model = regression_model
        logger.info("Se eligió el modelo de regresión lineal.")
    else:
        best_model = model_hgb
        logger.info("Se eligió el modelo HistGradientBoostingRegressor.")

    # Predicción por mes para los 12 meses del año usando el mejor modelo
    months = list(range(1, 13))  # Enero (1) hasta Diciembre (12)
    future_predictions = []
    for month in months:
        future_prediction = best_model.predict([[month, station_code]])[0]
        future_predictions.append(future_prediction)

    # Graficar las predicciones
    plt.figure(figsize=(12, 6))  # Aumentar el tamaño de la figura

    # Crear la gráfica de líneas
    plt.plot(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
              'Julio', 'Agosto', 'Septiembre', 'Octubre', 
              'Noviembre', 'Diciembre'], 
             future_predictions, marker='o', linestyle='-', color='cyan')

    # Configuración de etiquetas y título
    plt.xlabel('Meses', fontsize=14)
    plt.ylabel('Predicción de Velocidad del Viento (m/s)', fontsize=14)
    plt.title(f'Predicción Mensual de Velocidad del Viento para Estación {station_code}', fontsize=16)

    # Ajustar los límites del eje y para mejorar la visualización
    plt.ylim(0, max(future_predictions) * 1.1)  # Un poco más alto que el valor máximo

    # Rotar las etiquetas del eje x para evitar superposición
    plt.xticks(rotation=45)

    # Ajustar el diseño para que no se corten las etiquetas
    plt.tight_layout()

    # Asegúrate de que la carpeta exista antes de guardar la imagen.
    os.makedirs('static/graficas', exist_ok=True)

    plt.savefig(f'static/graficas/prediccion_{station_code}.png', bbox_inches='tight')  # Guardar con un nombre único
    plt.close()

    return {
        "station_code": station_code,
        "predicted_values": future_predictions,
        "months": ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 
                   'Junio', 'Julio', 'Agosto', 'Septiembre', 
                   'Octubre', 'Noviembre', 'Diciembre'],
        "image_url": f"/static/graficas/prediccion_{station_code}.png"  # URL para acceder a la imagen
    }
@app.get("/potencial_energia/{station_code}", tags=["Potencial Energía Eólica"])
def calcular_potencial_generacion(station_code: int):
    if wind_data_df.empty:
        raise HTTPException(status_code=500, detail="Los datos no están disponibles.")

    station_data = wind_data_df[wind_data_df['station_code'] == station_code]
    if station_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el código de estación: {station_code}")

    # Evaluar cuál modelo tiene mejor precisión (esto se puede ajustar según tu lógica)
    if regression_score > hgb_score:
        best_model = regression_model
        logger.info("Se eligió el modelo de regresión lineal.")
    else:
        best_model = model_hgb
        logger.info("Se eligió el modelo HistGradientBoostingRegressor.")

    # Predicción por mes para los 12 meses del año usando el mejor modelo
    months = list(range(1, 13))  # Enero (1) hasta Diciembre (12)
    future_predictions = []
    for month in months:
        future_prediction = best_model.predict([[month, station_code]])[0]
        future_predictions.append(future_prediction)

    # Calcular el potencial de generación de energía eólica (kW)
    # Utilizamos la fórmula P = 0.5 * ρ * A * v³, donde:
    # P es la potencia (W), ρ es la densidad del aire (aproximadamente 1.225 kg/m³ al nivel del mar),
    # A es el área barrida por las aspas del aerogenerador (m²), y v es la velocidad del viento (m/s).

    # Definimos un área típica para un aerogenerador (ejemplo: 100 m²)
    area_aerogenerador = 100  # m²
    densidad_aire = 1.225  # kg/m³

    potencia_kW = [(0.5 * densidad_aire * area_aerogenerador * (v ** 3)) / 1000 for v in future_predictions]

    # Graficar las predicciones y el potencial de generación
    plt.figure(figsize=(12, 6))

    # Gráfica de velocidad del viento
    plt.subplot(2, 1, 1)
    plt.plot(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 
              'Junio', 'Julio', 'Agosto', 'Septiembre', 
              'Octubre', 'Noviembre', 'Diciembre'], 
             future_predictions, marker='o', linestyle='-', color='cyan')
    
    plt.xlabel('Meses', fontsize=14)
    plt.ylabel('Velocidad del Viento (m/s)', fontsize=14)
    plt.title(f'Predicción Mensual de Velocidad del Viento para Estación {station_code}', fontsize=16)

    # Gráfica de potencia generada
    plt.subplot(2, 1, 2)
    plt.plot(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 
              'Junio', 'Julio', 'Agosto', 'Septiembre', 
              'Octubre', 'Noviembre', 'Diciembre'], 
             potencia_kW, marker='o', linestyle='-', color='green')
    
    plt.xlabel('Meses', fontsize=14)
    plt.ylabel('Potencial de Generación (kW)', fontsize=14)
    plt.title(f'Potencial Mensual de Generación Eólica para Estación {station_code}', fontsize=16)

    plt.tight_layout()

    # Asegúrate de que la carpeta exista antes de guardar la imagen.
    os.makedirs('static/graficas/potenciales', exist_ok=True)

    plt.savefig(f'static/graficas/potencial_{station_code}.png', bbox_inches='tight')
    plt.close()

    return {
        "station_code": station_code,
        "predicted_values": future_predictions,
        "potentials_kW": potencia_kW,
        "months": ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo',
                   'Junio', 'Julio', 'Agosto', 'Septiembre',
                   'Octubre', 'Noviembre', 'Diciembre'],
        "image_url": f"/static/graficas/potencial_{station_code}.png"  # URL para acceder a la imagen
    }

# Endpoint del chatbot
@app.post("/chat", response_class=HTMLResponse)
async def chat_response(request: Request, user_input: str = Form(...)):
    global coordenadas_usuario, municipio_usuario, nombre_usuario

    # Procesar entrada del usuario
    logger.info(f"Pregunta del usuario: {user_input}")

    # Inicializar las variables globales si aún no se han definido
    if 'nombre_usuario' not in globals() or nombre_usuario is None:
        nombre_usuario = None
    if 'municipio_usuario' not in globals() or municipio_usuario is None:
        municipio_usuario = None
    if 'coordenadas_usuario' not in globals() or coordenadas_usuario is None:
        coordenadas_usuario = None

    # Paso 1: Obtener el nombre del usuario si no está definido
    if nombre_usuario is None:
        nombre_usuario = user_input.strip().capitalize()
        response = f"Hola, {nombre_usuario}. ¿Cómo te puedo ayudar hoy? Para empezar, ¿te gustaría conocer tu ubicación?"

    # Paso 2: Obtener ubicación
    elif coordenadas_usuario is None and any(
        keyword in user_input.lower() for keyword in ["si", "claro", "ubicación", "localización", "donde estoy", "quiero saber mi ubicación", "sí, quiero"]
    ):
        try:
            coordenadas_usuario = obtener_ubicacion_usuario()
            response = f"Tus coordenadas son: {coordenadas_usuario[0]}, {coordenadas_usuario[1]}. ¿En qué municipio vives?"
        except Exception as e:
            logger.error(f"Error al obtener ubicación: {e}")
            response = "No se pudo obtener la ubicación. Intenta nuevamente."

    # Paso 3: Ingresar el municipio
    elif municipio_usuario is None:
        municipio_usuario = user_input.strip().lower()
        try:
            estaciones_info = obtener_estaciones_por_municipio(municipio_usuario)
            response = (
                f"Se encontraron {len(estaciones_info['estaciones'])} estaciones en {municipio_usuario.capitalize()}. "
                "¿Quieres saber si tu municipio tiene buenos vientos: (escribe clasificar (nombre municipio)"
            )
        except HTTPException as e:
            response = e.detail
        except Exception as e:
            logger.error(f"Error al buscar estaciones: {e}")
            response = f"No se pudieron encontrar estaciones en {municipio_usuario.capitalize()}. Intenta con otro municipio."

    # Paso 4: Clasificar el municipio
    elif "clasificar" in user_input.lower():
        try:
            partes = user_input.lower().split("clasificar", 1)
            if len(partes) < 2 or not partes[1].strip():
                response = "Por favor, ingresa el nombre del municipio después de la palabra 'clasificar'."
            else:
                municipio_usuario = partes[1].strip()
                clasificacion = clasificar_municipio(municipio_usuario)
                response = (
                    f"El municipio {municipio_usuario.capitalize()} se clasifica como: {clasificacion['classification']}. "
                    f"Cantidad de estaciones: {clasificacion['station_count']}, "
                    f"Velocidad promedio: {clasificacion['average_speed']:.2f} m/s. Se consideran buenas velocidades las que son cercanas a 8 m/s. "
                    "¿Quieres saber cuál estación está a menos de 5 km? (Responde 'sí' o 'no')"
                )
        except HTTPException as e:
            response = e.detail
        except Exception as e:
            logger.error(f"Error al clasificar municipio: {e}")
            response = "No se pudo realizar la clasificación del municipio. Inténtalo nuevamente."

    # Paso 5: Comparar distancias
    elif any(keyword in user_input.lower() and coordenadas_usuario and municipio_usuario for keyword in ["sí", "ok", "de acuerdo", "bien"]):
        try:
            estaciones_cercanas_info = obtener_estaciones_cercanas(municipio_usuario)
            if estaciones_cercanas_info["total_cercanas"] > 0:
                estaciones = estaciones_cercanas_info["estaciones_cercanas"]
                estaciones_list = "\n".join(
                    [f"- Estación {e['station_code']} a {e['distancia']:.2f} km" for e in estaciones]
                )
                response = (f"Estaciones cercanas dentro de un radio de 5 km:\n{estaciones_list}."   
                            "¿Quieres conocer la probabilidad de que los valores de viento medidos en las estaciones esten en tu ubicación?: (escribe probabilidad (nombre municipio))"
                            )
            else:
                estaciones_list = "\n".join(
                    [f"- Estación {e['station_code']} a {e['distancia']:.2f} km" for e in estaciones_cercanas_info["todas_estaciones"]]
                )
                response = (
                    "No se encontraron estaciones dentro de un radio de 5 km.\n"
                    f"Estaciones disponibles más cercanas:\n{estaciones_list}"
                    "¿Quieres conocer la probabilidad de que los valores de viento medidos en las estaciones esten en tu ubicación?: (escribe probabilidad (nombre municipio))"
                )
        except HTTPException as e:
            response = e.detail
        except Exception as e:
            logger.error(f"Error al buscar estaciones cercanas: {e}")
            response = "Ocurrió un error al calcular las estaciones cercanas."
    
    # Paso 6: Obtener probabilidad de viento
    elif user_input.lower().startswith("probabilidad"):
        try:
        # Verificar si el municipio está definido
            if municipio_usuario is None:
                response = "Primero debes ingresar tu municipio antes de consultar la probabilidad del viento."
            else:
                # Obtener la probabilidad de viento para el municipio actual
                probabilidad_resultado = await obtener_probabilidad_viento(municipio_usuario)
                
                # Verificar si hay resultados
                if not probabilidad_resultado["probabilidades_viento"]:
                    response = f"No se encontraron datos de probabilidad de viento para {municipio_usuario.capitalize()}."
                else:
                    # Construir la respuesta
                    probabilidades = probabilidad_resultado["probabilidades_viento"]
                    detalles = "".join(
                        f"Estación: {p['station_code']}<br>"
                        f"Distancia: {p['distancia']:.2f} km<br>"
                        f"Valor observado: {p['observed_value']:.2f}<br>"
                        f"Probabilidad: {p['probabilidad']:.2f}<br><br>"
                        for p in probabilidades
                    )
                    response = (
                        f"<strong>Probabilidad de viento en {municipio_usuario.capitalize()}:</strong><br>"
                        f"{detalles}"
                        "¿Quieres la predicción del viento en un año para una estación? (escribe 'proyectar (codigo de estacion)')."
                    )

        except HTTPException as e:
            response = e.detail
        except Exception as e:
            logger.error(f"Error al obtener probabilidad de viento: {e}")
            response = "Ocurrió un error al calcular la probabilidad del viento."

    # Paso 7: Predicción de velocidad del viento
    elif any(keyword in user_input.lower() for keyword in ["proyectar","predice", "cómo estará", "necesito saber", "predecir", "cómo va a estar", "predicción del futuro"]):
        try:
            partes = user_input.split()
            if len(partes) < 2 or not partes[-1].isdigit():
                response = "Por favor, ingresa un comando válido como 'predecir [código de estación]'."
            else:
                estacion_codigo = int(partes[-1])
                prediccion = predecir_velocidad_viento(estacion_codigo)
                
                valores_predichos = "\n".join(
                    [f"{mes}: {valor:.2f} m/s" for mes, valor in zip(prediccion['months'], prediccion['predicted_values'])]
                )
                
                response = (
                    f"<strong>Predicción de velocidad del viento para la estación {estacion_codigo}:</strong><br>"
                    f"{valores_predichos}<br>"
                    f"<img src='{prediccion['image_url']}' alt='Gráfica' style='max-width:100%; height:auto;'><br>"
                    f"Esta es la proyección para todo un año de la estación {estacion_codigo}"
                    "¿Quieres conocer el potencial de generación de energia eólica para esa estación?: escribe 'potencial (codigo de estacion)'"
                )
        
        except HTTPException as e:
            response = e.detail
        except ValueError:
            response = "Por favor, ingresa un código de estación válido como un número entero."
        except Exception as e:
            logger.error(f"Error al realizar la predicción: {e}")
            response = "No se pudo realizar la predicción. Asegúrate de ingresar un código de estación válido."

    # Paso 8: Obtener potencial energético
    elif any(keyword in user_input.lower() for keyword in ["potencial","si", "por favor", "okey"]):
        try:
            partes = user_input.split()
            if len(partes) < 2 or not partes[-1].isdigit():
                response = "Por favor, ingresa un comando válido como 'potencial [código de estación]'."
            else:
                estacion_codigo = int(partes[-1])
                potencial_resultado = calcular_potencial_generacion(estacion_codigo)
                
                valores_potenciales = "\n".join(
                    [f"{mes}: {valor:.2f} kW" for mes, valor in zip(potencial_resultado['months'], potencial_resultado['potentials_kW'])]
                )
                
                response = (
                    f"<strong>Potencial energético para la estación {estacion_codigo}:</strong><br>"
                    f"{valores_potenciales}<br>"
                    f"<img src='{potencial_resultado['image_url']}' alt='Gráfica' style='max-width:100%; height:auto;'><br>"
                    f"Este es el potencial energético proyectado para todo un año."
                )
        
        except HTTPException as e:
            response = e.detail
        except Exception as e:
            logger.error(f"Error al buscar estaciones cercanas: {e}")
            response = "Ocurrió un error al calcular las estaciones cercanas."

    

    # Respuesta para cualquier entrada desconocida
    else:
        response = "No entendí tu respuesta. Por favor, responde con una opción válida."

    return templates.TemplateResponse("chatbot.html", {"request": request, "response": response})