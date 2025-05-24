"""Con un csv con todas las columnas (o mínimo con estas: _id, review_scores, scores_media_pred),
este programa evalúa las predicciones realizadas por el modelo, calcula el error por cada propiedad,
y calcula la media de los errores, para determinar si el modelo ha hecho buenas predicciones.
Output: evaluacion_portugal_spain_trad_scores.csv (con las columnas _id, scores_media_real,
scores_media_pred y error. La media de los errores se imprime por la terminal. Si el error medio es
pequeño (cerca de 0), significa que el modelo hace buenas predicciones.
"""
import argparse
import ast
import os
import pandas as pd

#USO: python evaluacion.py --csv g_portugal_spain_trad_scores.csv
#     python evaluacion.py --csv t_portugal_spain_trad_scores.csv

parser=argparse.ArgumentParser(description='evaluacion')
parser.add_argument('--csv', type=str, help='csv a evaluar', required=True)
args=parser.parse_args()

df=pd.read_csv(args.csv)

ids=[]
scores_media_reales=[]
scores_media_predichos=[]
errores=[]

for n,fila in df.iterrows():
    #print("n: "+str(n))
    #print(fila["_id"])
    review_scores=fila["review_scores"]
    review_scores=ast.literal_eval(review_scores) #convertimos el string a diccionario
    scores_media_real=review_scores.get("review_scores_value") #obtenemos la media de scores real
    error=None
    if scores_media_real and ast.literal_eval(fila["reviews"]): #si scores_media_real no es None y hay reviews
        scores_media_real=scores_media_real*9/10 #convertimos la media real de 0 a 10 a de 0 a 9
        scores_media_pred=fila["scores_media_pred"] #obtenemos la media de scores predicha
        error=(scores_media_pred-scores_media_real)**2 #calculamos el error -> error=(media_predicha-media_real)^2
        ids.append(fila["_id"])
        scores_media_reales.append(scores_media_real)
        scores_media_predichos.append(scores_media_pred)
        errores.append(error)

print("De "+str(len(df))+" instancias, evaluamos "+str(len(ids))+" instancias, que son las que tienen el rating real y al menos una review.")

mse=sum(errores)/len(errores) #calculamos el error medio sin tener en cuenta los vacíos (mse=mean squared error)
print("MSE (error cuadrático medio): "+str(mse))
print("Scores media real: " + str(sum(scores_media_reales)/len(scores_media_reales)))
print("Scores media predicho: " + str(sum(scores_media_predichos)/len(scores_media_predichos)))

df_evaluacion=pd.DataFrame()
df_evaluacion["_id"]=ids
df_evaluacion["scores_media_real"]=scores_media_reales
df_evaluacion["scores_media_pred"]=scores_media_predichos
df_evaluacion["error"]=errores

#guardamos los resultados de la evaluacion en un csv
df_evaluacion.to_csv("evaluacion_"+os.path.splitext(args.csv)[0]+".csv",index=False)

print("El csv evaluacion_"+os.path.splitext(args.csv)[0]+".csv se ha guardado correctamente.")
