"""
"""
import argparse
import ast
import os
import pandas as pd

import Plantilla

#USO: python predecir_tradicional.py --csv portugal_spain_trad.csv --model modelo_random_forest_top1

parser=argparse.ArgumentParser(description='prediccion tradicionales')
parser.add_argument('--csv', type=str, help='csv a predecir (el csv tiene la columna:comments_traducidos)', required=True)
parser.add_argument('--model', type=str, help='nombre del modelo que queremos usar para predecir', required=True)
args=parser.parse_args()

#print(df.head())
df=pd.read_csv(args.csv)
scores_fila=[]
scores_columna=[]
scores_media_columna=[]

for n,fila in df.iterrows(): #por cada fila (en la que hay cero o varios comentarios)
    print("n: " + str(n))
    print(fila["_id"])
    comments=fila["comments_traducidos"]
    c = ast.literal_eval(comments) #en c hay una lista de comentarios
    df_c = pd.DataFrame(c) #en df_c hay un dataframe con los comentarios (un comentario en cada fila)
    if not df_c.empty and 0 in df_c.columns:
        df_c = df_c[~df_c[0].str.contains("This is an automatic message", na=False)] #si es un comentario automático, lo ignoramos porque
        df_c = df_c.reset_index(drop=True)                                           #no lo ha escrito un usuario, no expresa satisfacción o insatisfacción
    scores_fila=Plantilla.predecirScores(df_c, "comentarios.json", args.model)
    #print(scores_fila)
    scores_columna.append(scores_fila) #añadimos la lista de scores de una fila a la lista de scrores que representan la columna de scores (esto es, una lista dentro de otra lista)
    if len(scores_fila)>0:
        media=sum(scores_fila)/len(scores_fila) #sacamos la media
    else:
        media=None
    scores_media_columna.append(media) #añadimos la media a la lista
    scores_fila=[] #vaciamos la lista para la siguiente iteración

df["scores_pred"]=scores_columna #añadimos la lista de scrores final al df con los comentarios
df["scores_media_pred"]=scores_media_columna #añadimos la media de los scores

nombre_salida_csv="t_"+os.path.splitext(args.csv)[0]+"_scores.csv" #nombre del csv en el q se van a guardar los comentarios traducidos con scores

#guardamos en csv
df.to_csv(nombre_salida_csv,index=False)

print("El csv t_"+os.path.splitext(args.csv)[0]+"_scores.csv se ha guardado correctamente.")
#t_ para indicar que se han predicho los scores con un modelo tradicional entrenado por nosotras


