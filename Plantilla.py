# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import re
import string
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
from nltk import WordNetLemmatizer
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# Funciones auxiliares
# dev para evaluar sobre él. probamos con distintos hiperparámetros sobre dev. al final del tod. se prueba con el test para ver si algún algoritmo ha hecho overfitting
# semilla: para que el random sea "igual", importante para el ML, en la reproducibidad
def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-j", "--json", help="Archivo json (archivo.json)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)",
                       required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1,
                       type=int)
    parse.add_argument("--debug",
                       help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]",
                       required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()

    # Leemos los parametros del JSON
    with open(args.json) as json_file:
        preprocessing = json.load(json_file)

    # Juntamos tod. en una variable
    for key, value in preprocessing.items():
        setattr(args, key, value)

    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)  # fore es para la consola
        return data
    except Exception as e:
        print(Fore.LIGHTMAGENTA_EX + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:,
                              categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]

        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + "> Columnas numéricas:\n" + Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA + "> Columnas de texto:\n" + Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA + "> Columnas categóricas:\n" + Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.LIGHTMAGENTA_EX + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.
    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.
    Returns:
        None
    Raises:
        None

    """
    global data
    try:
        for c in numerical_feature.columns:
            if 'impute' in args.preprocessing["missing_values"]:
                #print("----------------------'impute' in args.preprocessing[]")
                if 'mean' in args.preprocessing["impute_strategy"]:
                    #print("--------------------- HA ENTRADO-------------------")
                    media = numerical_feature[c].mean()
                    numerical_feature[c].fillna(media, inplace=True)
                elif 'median' in args.preprocessing["impute_strategy"]:
                    mediana = numerical_feature[c].median()
                    numerical_feature[c].fillna(mediana, inplace=True)
                elif 'mode' in args.preprocessing["impute_strategy"]:
                    moda = numerical_feature[c].mode()[0]
                    numerical_feature[c].fillna(moda, inplace=True)
            elif 'drop' in args.preprocessing["missing_values"]:
                numerical_feature[c].dropna(inplace=True)  # inplace=true es que te lo modifica, no te hace una copia
        for c in categorical_feature.columns:
            if 'impute' in args.preprocessing["missing_values"]:
                moda = categorical_feature[c].mode()[0]
                categorical_feature[c].fillna(moda, inplace=True)
            elif 'drop' in args.preprocessing["missing_values"]:
                categorical_feature[c].dropna(inplace=True)

        if 'impute' in args.preprocessing["missing_values"]: #si impute si o si va a ser la moda para este campo xq o 1 o 0
            moda = data[args.prediction].mode()[0]
            data[args.prediction].fillna(moda, inplace=True)
        elif 'drop' in args.preprocessing["missing_values"]:
            # print("----------------------'drop' in args.preprocessing[] PARA PREDICTION")
            #args.prediction[c].dropna(inplace=True)
            data.dropna(subset=[args.prediction], inplace=True)

        data[numerical_feature.columns] = numerical_feature  # Actualiza columnas numéricas en 'data'
        data[categorical_feature.columns] = categorical_feature
        print(Fore.LIGHTMAGENTA_EX + "Valores faltantes tratados correctamente :)" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al procesar valores faltantes" + Fore.RESET)
        print(e)
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.
    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.
    Returns:
        None
    Raises:
        Exception: Si hay un error al reescalar los datos.
    """
    global data
    try:
        scaler = None
        if "max" in args.preprocessing["scaling"]:
            scaler = MaxAbsScaler()
        elif "minmax" in args.preprocessing["scaling"]:
            scaler = MinMaxScaler()
        elif "standard" in args.preprocessing["scaling"]:
            scaler = StandardScaler()
        elif "normalize" in args.preprocessing["scaling"]:
            scaler = Normalizer()
        if scaler:
            # Convertir booleanos a float para evitar problemas con el escalador
            #data[numerical_columns] = data[numerical_columns].astype(float)
            numerical_feature = numerical_feature.astype(float)
            numerical_feature[:] = scaler.fit_transform(numerical_feature)

            data[numerical_feature.columns] = numerical_feature
            print(Fore.LIGHTMAGENTA_EX + "Datos correctamente escalados :)" + Fore.RESET)
            print(data.head)

    except Exception as e:
        print(Fore.RED + "Error al escalar" + Fore.RESET)
        print(e)
        sys.exit(1)

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.
    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.
    """
    global data
    try:
        for c in categorical_feature.columns:
            # Comprobamos si la columna es de tipo 'object' (categórica)
            if categorical_feature[c].dtype == 'object':
                categorical_feature[c] = LabelEncoder().fit_transform(categorical_feature[c])
                data[c] = categorical_feature[c]
        print(Fore.LIGHTMAGENTA_EX + "cat2num correcto :)" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al convertir características categóricas" + Fore.RESET)
        print(e)
        sys.exit(1)


def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....
    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    Retorna:
    None
    """
    global data
    try:
        if text_feature.empty:
            print(Fore.YELLOW + "No hay datos de texto para simplificar." + Fore.RESET)
            return
        for c in text_feature.columns:  # por cada columna
            # txt2minúsculas
            if args.preprocessing["text_processing"].get("lowercase", False):
                text_feature[c] = text_feature[c].apply(lambda x: x.lower() if isinstance(x, str) else x)

            if args.preprocessing["text_processing"].get("remove_punctuation", False):
                text_feature[c] = text_feature[c].apply( lambda x: re.sub(f"[{string.punctuation}]", "", x) if isinstance(x, str) else x)

            # tokenizamos
            if args.preprocessing["text_processing"].get("tokenization", False): # si en el json no está tokenization devuelve false
                text_feature[c] = text_feature[c].apply(
                    lambda x: word_tokenize(x) if isinstance(x, str) and x.strip() != "" else [])
            else:  # IMPORTANTE: stemmer y lemmatizer se aplican sobre listas de tokens, por eso está el else para que haga split sin más
                text_feature[c] = text_feature[c].apply(
                    lambda x: x.split() if isinstance(x, str) and x.strip() != "" else [])
            # el strip() es por si tenemos celdas vacías o con solo espacios en blanco, para que las elimine y no nos de errores en lo de stopwords

            # eliminamos stopwords
            if args.preprocessing["text_processing"].get("stopwords", False):
                stop_words = set(stopwords.words('spanish'))
                text_feature[c] = text_feature[c].apply(
                    lambda x: [word for word in x if word not in stop_words] if isinstance(x, list) else x)

            # steam: raíz: correr, corriendo -> corr
            if args.preprocessing["text_processing"].get("stemmer", False):
                stemmer = PorterStemmer()
                text_feature[c] = text_feature[c].apply(
                    lambda x: [stemmer.stem(word) for word in x] if isinstance(x, list) else x)
            # si no, miramos a ver si queremos lematizar: correr, corriendo -> correr
            elif args.preprocessing["text_processing"].get("lemmatization", False):
                lemmatizer = WordNetLemmatizer()
                text_feature[c] = text_feature[c].apply(
                    lambda x: [lemmatizer.lemmatize(word) for word in x] if isinstance(x, list) else x)
            if args.preprocessing["text_processing"].get("sort", False):
                text_feature[c] = text_feature[c].apply(lambda x: sorted(x) if isinstance(x, list) else x)
            data[c] = text_feature[c]
        print(Fore.LIGHTMAGENTA_EX + "Texto simplificado :)" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al simplificar texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if "tf-idf" in args.preprocessing["text_processing"]["method"]:
                tfidf_vectorizer = TfidfVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                #print(tfidf_vectorizer.get_feature_names_out())
                text_features_df = pd.DataFrame(tfidf_matrix.toarray(),
                                                columns=tfidf_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.LIGHTMAGENTA_EX + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)

            elif "bow" in  args.preprocessing["text_processing"]["method"]:
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                #print(bow_vecotirizer.get_feature_names_out())
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.LIGHTMAGENTA_EX + "Texto tratado con éxito usando BOW" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al procesar texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    Args:
        None
    Returns:
        None
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    global data
    try:
        X = data.drop(columns=[args.prediction])
        y = data[args.prediction]
        if (args.preprocessing["sampling"]["percent"] == 'auto'):
            percent = "auto"
        else:
            percent = calculoPorcentaje(args.preprocessing["sampling"]["percent"], y)
        print("---->Distribución de clases antes del sampling:")
        print(y.value_counts())  # clases originales antes de balancear

        if "oversampling" in args.preprocessing["sampling"]["type"]:
            sampler = RandomOverSampler(sampling_strategy=percent, random_state=42)
        else:  # "undersampling" in args.preprocessing["sampling"]["type"]:
            sampler = RandomUnderSampler(sampling_strategy=percent, random_state=42)

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        print("Distribución de clases después del sampling:")
        print(y_resampled.value_counts())  # Muestra las clases balanceadas después del proceso

        # Reunir X_resampled e y_resampled en un único DataFrame
        data = pd.concat([X_resampled, y_resampled], axis=1)
        print("Datos balanceados:")
        print(data.head())
    except Exception as e:
        print(Fore.RED + "Error al realizar el oversampling/undersampling." + Fore.RESET)
        print(e)
        sys.exit(1)

def calculoPorcentaje(p, y): #esto es porq si es clase binaria 0.5 vale pero si es una multiclase es como q hay q usar un "diccionario"
    numClases = y.value_counts()
    if "oversampling" in args.preprocessing["sampling"]["type"]:
        if len(numClases) > 2:  # Multiclase
            max_clase = numClases.max()  # Renombramos "max"
            return {cls: max(numClases[cls], int(max_clase * p)) for cls in numClases.index if
                    numClases[cls] < max_clase}
        else:  # Binario
            return p

    else:  # Undersampling
        if len(numClases) > 2:  # Multiclase
            min_clase = numClases.min()
            return {cls: min(numClases[cls], int(min_clase * p)) for cls in numClases.index if
                    numClases[cls] > min_clase}
        else:  # Binario
            return p
def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.LIGHTMAGENTA_EX+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    global data

    numerical_feature, text_feature, categorical_feature = select_features()

    # Simplificamos el texto
    if "text_processing" in args.preprocessing:
        simplify_text(text_feature)
        process_text(text_feature)


    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)
    print("------------------DESPUES CAT2NUM-----------------")
    print(data.head())
    print("----------------------------------------------------------------------------------------------------")
    # Tratamos missing values
    if "missing_values" in args.preprocessing:
        process_missing_values(numerical_feature, categorical_feature)
        print("------------------DESPUES MISSING VALUES-----------------")
        print(data.head())
        print("----------------------------------------------------------------------------------------------------")

    # Reescalamos los datos numéricos
    if "scaling" in args.preprocessing:
        reescaler(numerical_feature)
        print("------------------DESPUES SCALING-----------------")
        print(data.head())
        print("----------------------------------------------------------------------------------------------------")

    #data[numerical_feature.columns] = numerical_feature
    #data[categorical_feature.columns] = categorical_feature

    # Realizamos Oversampling o Undersampling
    if "sampling" in args.preprocessing:
        over_under_sampling()
        print("------------------DESPUES UNDERSAMPLING/OVERSAMPLING-----------------")
        print(data.head())
        print("----------------------------------------------------------------------------------------------------")

    if "drop_features" in args.preprocessing:
        drop_features()
    #por alguna razón sin estas líneas de abajo no se reflejan los cambios y encima se duplican las cosas sin más eh

    return pd.concat([numerical_feature, text_feature, categorical_feature, data[args.prediction]], axis=1)

# Funciones para entrenar un modelo

 
 
def save_model(gs, nombrealgoritmo):
    """
    Guarda el modelo (o los modelos) y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.
    - nombrealgoritmo: nombre del algoritmo

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        results = gs.cv_results_
        #print(results)
        #creamos los nombres de las métricas a guardar (por ejemplo, mean_test_accuracy, mean_test_f1_macro, etc.)
        metric_columns = [f'mean_test_{metric}' for metric in args.evaluation]

        meantestx = f'mean_test_{args.best_model}' #respecto a q los ordenamos
        #odenamos la combinación de parámetros respecto a best_model (la métrica q hemos elegido)
        indices_ordenados = sorted(range(len(results[meantestx])), key=lambda i: results[meantestx][i], reverse=True)

        #creamos el csv
        with open(f'output/{nombrealgoritmo}-modelos.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Params'] + args.evaluation)
            #para combinación guardamos los parámetros y el valor de la métrica elegida
            for i in indices_ordenados:
                params = results['params'][i]
                scores = [results[metric][i] for metric in metric_columns]  #extraemos todas las métricas
                writer.writerow([params] + scores)

        #guardamos los x mejores modelos
        for rank, i in enumerate(indices_ordenados[:args.nummodelos]):
            params = results['params'][i]
            #entrenar el modelo i con los parámetros
            modelo = gs.best_estimator_.set_params(**params)
            with open(f'output/modelo_{nombrealgoritmo}_top{rank+1}.pkl', 'wb') as file:
                pickle.dump(modelo, file) #guardamos el modelo

        print(Fore.CYAN + f"Los {args.nummodelos} mejores modelos fueron guardados con éxito." + Fore.RESET)
        print(Fore.CYAN + "Todas las combinaciones de hiperparámetros se guardaron en algoritmo-modelos.csv" + Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)


def kNN(x_traindev, y_traindev):
    """
    class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
    leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    """
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param x_traindev, y_traindev: Conjunto de datos para realizar la clasificación.
    :type x_traindev, y_traindev: pandas.DataFrame
    :return: nada
    """

    #print(args.evaluation)
    #print(args.best_model)

    # en caso de tener en el json k_min, k_max, p_min y p_max actualizamos el args para pasarle un rango
    if "k_min" in args.kNN and "k_max" in args.kNN:
        args.kNN["n_neighbors"] = list(range(args.kNN["k_min"],args.kNN["k_max"]+1))
        del args.kNN["k_min"]
        del args.kNN["k_max"]
    if "p_min" in args.kNN and "p_max" in args.kNN:
        args.kNN["p"] = list(range(args.kNN["p_min"],args.kNN["p_max"]+1))
        del args.kNN["p_min"]
        del args.kNN["p_max"]

    """cv=5 indica que se usará una validación cruzada de 5 folds. Significa que los datos se dividirán
      en 5 subconjuntos: en cada iteración, 4 de ellos se usarán para entrenar y 1 para evaluar, y este proceso
      se repetirá 5 veces cambiando el subconjunto de evaluación. Es útil para evitar overfitting y obtener 
      una estimación más robusta del rendimiento del modelo."""

    gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=args.cpu, scoring=args.evaluation, refit=args.best_model)
    start_time = time.time()
    #entrenamos con traindev
    gs.fit(x_traindev, y_traindev)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    #guardamos el modelo utilizando pickle
    save_model(gs,'kNN')
    obtener_probabilidades()

def decision_tree(x_traindev, y_traindev):
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param x_traindev, y_traindev: Conjunto de datos para realizar la clasificación.
    :type x_traindev, y_traindev: pandas.DataFrame
    """

    #hacemos un barrido de hiperparametros
    gs = GridSearchCV(DecisionTreeClassifier(),param_grid= args.decisionTree, cv=5, n_jobs=args.cpu, scoring=args.evaluation, refit=args.best_model)
    start_time = time.time()
    gs.fit(x_traindev, y_traindev)
    end_time = time.time()
    execution_time = end_time - start_time

    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")

    """print(" RESULTADOS DECISION TREE")
    print(gs.cv_results_)"""

    #guardamos el modelo utilizando pickle
    save_model(gs, 'decision_tree')
    obtener_probabilidades()

def random_forest(x_traindev, y_traindev):
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    :param x_traindev, y_traindev: Conjunto de datos para realizar la clasificación.
    :type x_traindev, y_traindev: pandas.DataFrame
    """

    gs = GridSearchCV(RandomForestClassifier(), param_grid= args.randomForest, cv=5, n_jobs=args.cpu, scoring=args.evaluation, refit=args.best_model)
    start_time = time.time()
    gs.fit(x_traindev, y_traindev)
    end_time = time.time()
    execution_time = end_time - start_time

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    """print(" RESULTADOS DECISION TREE")
    print(gs.cv_results_)"""

    #guardamos el modelo utilizando pickle
    save_model(gs, 'random_forest')
    obtener_probabilidades()

def naive_bayes(x_traindev, y_traindev):
    gs = GridSearchCV(GaussianNB(), param_grid= args.naiveBayes, cv=5, n_jobs=args.cpu, scoring=args.evaluation, refit=args.best_model)
    start_time = time.time()
    gs.fit(x_traindev, y_traindev)
    end_time = time.time()
    execution_time = end_time - start_time

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    """print(" RESULTADOS NAIVE BAYES")
    print(gs.cv_results_)"""

    #guardamos el modelo utilizando pickle
    save_model(gs, 'naive_bayes')
    obtener_probabilidades()

def load_model():
    """
    Carga el modelo desde el archivo 'output/algoritmo-m.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo_algoritmo_topX.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(f'output/{args.nombremodeloatestear}.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)

def obtener_probabilidades():

    try:
        loaded_knn_model = load_model()
        print("Modelo k-NN cargado exitosamente.")

        print("\nCalculando probabilidades en el conjunto de test con el modelo cargado...")
        probabilities_loaded = loaded_knn_model.predict_proba(x_traindev)

        print("Clases del modelo cargado:", loaded_knn_model.classes_)
        print("Forma del array de probabilidades:", probabilities_loaded.shape)
        print("Probabilidades para las primeras 5 muestras de test (modelo cargado):")
        print(probabilities_loaded[:5])

    except FileNotFoundError:
        print(
            Fore.RED + "Error: No se encontró el archivo del modelo 'kNN_best_model.joblib'. Ejecuta la función kNN primero." + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"Error al cargar o usar el modelo: {e}" + Fore.RESET)

def calcular_metricas(y_real, y_pred):
    resultados = {}

    #calculamos las métricas si están en args.evaluation
    if "accuracy" in args.evaluation: resultados["accuracy"] = accuracy_score(y_real, y_pred)

    if "precision" in args.evaluation: resultados["precision"] = precision_score(y_real, y_pred, average="binary")
    if "recall" in args.evaluation: resultados["recall"] = recall_score(y_real, y_pred, average="binary")
    if "f1" in args.evaluation: resultados["f1"] = f1_score(y_real, y_pred, average="binary")
    if "evalaux" in args and "specificity" in args.evalaux: #solo en binarios
        tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
        if tn + fp > 0: #evitamos la división entre 0
            resultados["specificity"] = tn / (tn + fp)
        else:
            resultados["specificity"] = 0.0  #si no hay negativos, asignamos 0

    if "precision_micro" in args.evaluation: resultados["precision_micro"] = precision_score(y_real, y_pred, average="micro")
    if "recall_micro" in args.evaluation: resultados["recall_micro"] = recall_score(y_real, y_pred, average="micro")
    if "f1_micro" in args.evaluation: resultados["f1_micro"] = f1_score(y_real, y_pred, average="micro")

    if "precision_macro" in args.evaluation: resultados["precision_macro"] = precision_score(y_real, y_pred, average="macro")
    if "recall_macro" in args.evaluation: resultados["recall_macro"] = recall_score(y_real, y_pred, average="macro")
    if "f1_macro" in args.evaluation: resultados["f1_macro"] = f1_score(y_real, y_pred, average="macro")

    if "precision_weighted" in args.evaluation: resultados["precision_weighted"] = precision_score(y_real, y_pred, average="weighted")
    if "recall_weighted" in args.evaluation: resultados["recall_weighted"] = recall_score(y_real, y_pred, average="weighted")
    if "f1_weighted" in args.evaluation: resultados["f1_weighted"] = f1_score(y_real, y_pred, average="weighted")

    #convertimos el diccionario de resultados a un DataFrame
    return pd.DataFrame(resultados, index=[0])
    
# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    global data
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            dataname=os.path.splitext(os.path.basename(args.file))[0] #si el file es iris.csv nos quedamos con iris (sin el .csv)
            data.to_csv(f'output/{dataname}-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)

    #dividimos los datos en train, dev y test
    x = data.drop(columns=[args.prediction])  # Todas las columnas excepto la target
    y = data[args.prediction]  # Solo la columna target
    #dividimos los datos en 2: traindev y test dependiendo al porcentaje de test
    x_traindev, x_test, y_traindev, y_test = train_test_split(x, y, test_size=args.porcentajetest, stratify=y)

    #ejemplo: porcentajetest=0.25 -> data se divide en: traindev 75% y test 25%
    #         luego le pasamos a gridsearch traindev y él internamente divide los datos para entrenar y evaluar

    # Cantidad de datos en cada conjunto
    print(f"Cantidad de datos en traindev: {len(x_traindev)}")
    print(f"Cantidad de datos en test: {len(x_test)}")
    # Contar instancias de cada clase en traindev
    print("\nDistribución de clases en traindev:")
    print(y_traindev.value_counts())
    # Contar instancias de cada clase en test
    print("\nDistribución de clases en test:")
    print(y_test.value_counts())

    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN(x_traindev, y_traindev)
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree(x_traindev, y_traindev)
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest(x_traindev, y_traindev)
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "naive_bayes":
            try:
                naive_bayes(x_traindev, y_traindev)
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            # Predecimos
            y_pred = model.predict(x_test)

            """print("\nX-TEXT -----------------------------------")
            print(x_test)
            print("\nY-TEXT -----------------------------------")
            print(y_test)
            print("\nY-PRED -----------------------------------")
            print(y_pred)"""

            cm=confusion_matrix(y_test, y_pred)
            print("\nMATRIZ DE CONFUSIÓN:")
            print(cm)
            #obtener las etiquetas únicas de las clases
            labels = sorted(set(y_test))  #aseguramos el orden correcto
            #crear DataFrame con etiquetas en filas y columnas
            df_cm = pd.DataFrame(cm, index=[f'Real-{c}' for c in labels], columns=[f'Pred-{c}' for c in labels])
            df_cm.to_csv(f'output/{args.algorithm}-confusion_matrix.csv', index=True) #guardar como CSV

            cr = classification_report(y_test, y_pred, output_dict=True)  #convertir a diccionario
            print("\nCLASSIFICATION REPORT:")
            print(classification_report(y_test, y_pred))
            df_cr = pd.DataFrame(cr).transpose()  #convertir a DataFrame
            df_cr.to_csv(f'output/{args.algorithm}-classification_report.csv', index=True)  #guardar como CSV


            # Añadimos la prediccion al dataframe data
            datatest = pd.concat([x_test.reset_index(drop=True), y_test.rename(args.prediction + '-real').reset_index(drop=True), pd.DataFrame(y_pred, columns=[args.prediction+'-prediccion']).reset_index(drop=True)], axis=1)

            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            datatest.to_csv(f'output/test-prediction-{args.nombremodeloatestear}.csv', index=False)
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)

            dfmetricas=calcular_metricas(y_test, y_pred)
            dfmetricas.to_csv(f'output/test-metricas-{args.nombremodeloatestear}.csv', index=False)

            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
