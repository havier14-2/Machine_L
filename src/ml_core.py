import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore') # Para evitar advertencias visuales de K-Means
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def preparar_datos(ruta_archivo, umbral_nulos=0.4):
    """Carga, limpia, divide, preprocesa y balancea los datos."""
    # 1. Carga y limpieza inicial
    df = pd.read_excel(ruta_archivo)
    df = df.dropna(thresh=df.shape[0] * (1 - umbral_nulos), axis=1)
    
    X = df.drop(['Id', 'target'], axis=1) 
    y = df['target']
    
    # 2. División Estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Guardamos los índices originales de X_test para la proyección de negocio
    indices_test = X_test.index
    
    # 3. Pipelines
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='SinDato')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 4. Transformación
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # 5. Balanceo con SMOTE (Solo en Train)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)
    
    return X_train_res, y_train_res, X_test_prep, y_test, preprocessor, df, indices_test

def entrenar_modelos(X_train, y_train):
    """Entrena y retorna los 3 modelos supervisados solicitados."""
    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        
    return modelos

def evaluar_modelos(modelos, X_test, y_test):
    """Calcula y retorna un DataFrame con las métricas de negocio."""
    resultados = []
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        resultados.append({
            'Modelo': nombre,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    return pd.DataFrame(resultados).set_index('Modelo')

def mostrar_matriz_confusion(modelo, X_test, y_test, nombre_modelo):
    """Genera la gráfica de la matriz de confusión analizada."""
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Acepta (0)', 'Sí Acepta (1)'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Matriz de Confusión: {nombre_modelo}', fontsize=14)
    plt.grid(False) # Quitar cuadrícula de Seaborn para que se vea limpio
    plt.show()

def generar_proyeccion_negocio(modelo_final, preprocessor, df_original, X_test_prep, indices_test):
    """Identifica clientes potenciales y genera proyecciones comerciales."""
    # Predecir sobre el test set
    y_pred_proba = modelo_final.predict_proba(X_test_prep)[:, 1] # Probabilidad de clase 1
    y_pred_class = modelo_final.predict(X_test_prep)
    
    # Recuperar la información original de los clientes en la muestra de prueba
    df_test = df_original.loc[indices_test].copy()
    df_test['Probabilidad_Aceptacion'] = y_pred_proba
    df_test['Prediccion_Modelo'] = y_pred_class
    
    # Filtrar solo los potenciales clientes (los que el modelo dice que SÍ aceptarán)
    clientes_potenciales = df_test[df_test['Prediccion_Modelo'] == 1].sort_values(by='Probabilidad_Aceptacion', ascending=False)
    
    return clientes_potenciales



def evaluar_clusters_optimos(X_prep, max_clusters=10):
    """
    (Cumple Indicador 5)
    Calcula la Inercia (Método del Codo) y el Silhouette Score 
    para determinar el número ideal de clusters (K).
    """
    inercias = []
    siluetas = []
    
    RANGO_K = range(2, max_clusters + 1)
    
    for k in RANGO_K:
        # Se usa random_state para replicabilidad y n_init automático por buenas prácticas
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_prep)
        
        inercias.append(kmeans.inertia_)
        score_silueta = silhouette_score(X_prep, kmeans.labels_)
        siluetas.append(score_silueta)
        
    return RANGO_K, inercias, siluetas

def graficar_metricas_clustering(RANGO_K, inercias, siluetas):
    """
    (Cumple Indicador 2 y 5)
    Genera un gráfico dual para visualizar el Codo y la Silueta.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Gráfico del Método del Codo (Inercia)
    color = '#00205B'
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inercia (Suma de errores al cuadrado)', color=color, fontsize=12)
    ax1.plot(RANGO_K, inercias, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    # Gráfico del Silhouette Score
    ax2 = ax1.twinx()  
    color = '#E3D2A1'
    ax2.set_ylabel('Silhouette Score (Cercano a 1 es mejor)', color='#9ca3af', fontsize=12)  
    ax2.plot(RANGO_K, siluetas, marker='s', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor='#9ca3af')

    fig.tight_layout()  
    plt.title('Evaluación de Clusters Óptimos (Codo vs Silueta)', fontsize=14, fontweight='bold')
    plt.xticks(RANGO_K)
    plt.show()

def entrenar_modelo_no_supervisado(X_prep, k_optimo):
    """
    (Cumple Indicador 4 y 6)
    Entrena el modelo final de K-Means con el número K seleccionado.
    """
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init='auto')
    etiquetas = kmeans_final.fit_predict(X_prep)
    return kmeans_final, etiquetas






def perfilar_clusters(df_original, indices_test, etiquetas):
    """
    (Cumple Indicador 7 y 8)
    Une las etiquetas del clustering con los datos originales de prueba
    para interpretar el perfil de negocio de cada segmento.
    """
    import pandas as pd
    
    # Rescatamos a los 10,225 clientes reales con sus datos originales (sin estandarizar)
    df_test_real = df_original.iloc[indices_test].copy()
    
    # Les pegamos la etiqueta del grupo que el K-Means decidió
    df_test_real['Segmento_Asignado'] = etiquetas
    
    # Calculamos cuántos clientes quedaron en cada segmento
    volumen_segmentos = df_test_real['Segmento_Asignado'].value_counts().sort_index().rename("Cantidad de Clientes")
    
    # Calculamos el promedio de las variables numéricas para entender quiénes son
    cols_numericas = df_test_real.select_dtypes(include=['float64', 'int64']).columns
    perfilamiento = df_test_real.groupby('Segmento_Asignado')[cols_numericas].mean().T
    
    return df_test_real, perfilamiento, volumen_segmentos