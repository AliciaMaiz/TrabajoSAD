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





(iratxe)
airbnbCreateDatasetsForPojectAlumno.py para obtener portugal.csv, spain.csv y portugal_spain.csv
traduccion.py para obtener una columna con los comentarios traducidos
predecir_algoritmo_generativo.py para obtener las columnas scores y scores_media y para obtener el csv completo


