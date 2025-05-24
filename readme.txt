Ana Victoria Cernatescu, Maira Gabriela Herbas, Alicia Maizkurrena e Iratxe Urrutia Sordo

Para instalar dependencias:
    pip install -r requirements.txt

Para entrenar:
    python Plantilla.py -m train -f archivo.csv -j archivo.json -a algoritmo -p target --debug
Para testear:
    python Plantilla.py -m test -f archivo.csv -j archivo.json -a algoritmo -p target --debug

Información sobre el comando:
    -m, --mode: Modo de ejecución (train o test)
    -f, --file: Fichero csv (/Path_to_file)
    -j, --json: Archivo json (archivo.json)
    -a, --algorithm: Algoritmo a ejecutar (kNN, decision_tree o random_forest)
    -p, --prediction: Columna a predecir (Nombre de la columna)
    --debug: Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]


Extraer los datos de portugal y españa de airbnb.csv (output: portugal.csv, spain.csv y portugal_spain.csv):
    python airbnbCreateDatasetsForPojectAlumno.py

Teniendo portugal_spain.csv, para traducir los comentarios (output:portugal_spain_trad.csv):
    python traduccion.py --csv portugal_spain.csv

Teniendo portugal_spain_trad.csv, para predecir los ratings con el modelo generativo gemma2:2b (output: g_portugal_spain_trad_scores.csv):
    python predecir_algoritmo_generativo.py --csv portugal_spain_trad.csv

Teniendo portugal_spain_trad.csv, un modelo .pkl y un vectorizer .pkl, para predecir los ratings con ese modelo entrenado (output: t_portugal_spain_trad_scores.csv):
    python predecir_tradicional.py --csv portugal_spain_trad.csv --model modelo_{algoritmo}_topX

Teniendo g_portugal_spain_trad_scores.csv o t_portugal_spain_trad_scores.csv, para evaluar el modelo y calcular los errores:
    python evaluacion.py --csv g_portugal_spain_trad_scores.csv
    o
    python evaluacion.py --csv t_portugal_spain_trad_scores.csv
