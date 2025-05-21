"""
"""

import ast
import os
import pandas as pd

import Plantilla

nombre_csv="portugal_trad.csv" #csv a predecir (el csv tiene la columna:comments_traducidos)
#en cada fila del csv (en comments_traducidos) hay una lista de comentarios traducidos que pertenecen a una propiedad de airbnb

#print(df.head())
df=pd.read_csv(nombre_csv)
scores_fila=[]
scores_columna=[]
scores_media_columna=[]

for n,fila in df.iterrows(): #por cada fila (en la que hay cero o varios comentarios)
    print("n: " + str(n))
    print(fila["_id"])
    comments=fila["comments_traducidos"]
    c = ast.literal_eval(comments) #en c hay una lista de comentarios
    df_c = pd.DataFrame(c) #en df_c hay un dataframe con los comentarios (un comentario en cada fila)
    scores_fila=Plantilla.predecirScores(df_c, "comentarios.json","modelo_naive_bayes_top1")
    #print(scores_fila)
    scores_columna.append(scores_fila) #añadimos la lista de scores de una fila a la lista de scrores que representan la columna de scores (esto es, una lista dentro de otra lista)
    if len(scores_fila)>0:
        media=sum(scores_fila)/len(scores_fila) #sacamos la media
        #media=round(media)
    else:
        media=None
    scores_media_columna.append(media) #añadimos la media a la lista
    scores_fila=[] #vaciamos la lista para la siguiente iteración

df["scores_pred"]=scores_columna #añadimos la lista de scrores final al df con los comentarios
df["scores_media_pred"]=scores_media_columna #añadimos la media de los scores

nombre_salida_csv=os.path.splitext(nombre_csv)[0]+"_scores.csv" #nombre del csv en el q se van a guardar los comentarios traducidos con scores

#guardamos en csv
df.to_csv(nombre_salida_csv,index=False)

print("El csv "+os.path.splitext(nombre_csv)[0]+"_scores.csv se ha guardado correctamente.")



