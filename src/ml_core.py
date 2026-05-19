import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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