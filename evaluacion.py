"""Con un csv con todas las columnas (o mínimo con estas: _id, review_scores, scores_media_pred),
este programa evalúa las predicciones realizadas por el modelo, calcula el error por cada propiedad,
y calcula la media de los errores, para determinar si el modelo ha hecho buenas predicciones.
Output: evaluacion_portugal_spain_trad_scores.csv (con las columnas _id, scores_media_real,
scores_media_pred y error. La media de los errores se imprime por la terminal. Si el error medio es
pequeño (cerca de 0), significa que el modelo hace buenas predicciones.
"""

import ast
import os
import pandas as pd

nombre_csv="portugal_spain_trad_scores.csv"
df=pd.read_csv(nombre_csv)

scores_media_reales=[]
errores=[]

for n,fila in df.iterrows():
    print("n: "+str(n))
    print(fila["_id"])
    review_scores=fila["review_scores"]
    review_scores=ast.literal_eval(review_scores) #convertimos el string a diccionario
    scores_media_real=review_scores.get("review_scores_value") #obtenemos la media de scores real
    scores_media_reales.append(scores_media_real)
    error=None
    if scores_media_real: #si no es None
        scores_media_real=scores_media_real*9/100 #convertimos la media real de 0 a 100 a de 0 a 9
        scores_media_pred=fila["scores_media_pred"] #obtenemos la media de scores predicha
        error=(scores_media_pred-scores_media_real)**2 #calculamos el error -> error=(media_predicha-media_real)^2
    errores.append(error)

errores_validos=[e for e in errores if e is not None]
mse=sum(errores_validos)/len(errores_validos) #calculamos el error medio sin tener en cuenta los vacíos (mse=mean squared error)
print("MSE (error cuadrático medio): "+str(mse))

df_evaluacion=pd.DataFrame()
df_evaluacion["_id"]=df["_id"]
df_evaluacion["scores_media_real"]=scores_media_reales
df_evaluacion["scores_media_pred"]=df["scores_media_pred"]
df_evaluacion["error"]=errores

#guardamos los resultados de la evaluacion en un csv
df_evaluacion.to_csv("evaluacion_"+os.path.splitext(nombre_csv)[0]+".csv",index=False)

print("El csv evaluacion_"+os.path.splitext(nombre_csv)[0]+".csv se ha guardado correctamente.")
