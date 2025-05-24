"""Con portugal_spain_trad.csv, este programa obtiene los comentarios de cada propiedad, y por cada comentario predice
el score del usuario basándose en su comentario. Por cada propiedad que contenga una lista de comentarios, se crea una
lista de scores predichos, y se calcula la media de los scores predichos. Tanto la lista de scores como la media se
añaden al csv que contiene todos los datos de portugal y españa (incluyendo la columna con los comentarios traducidos).
-Output: portugal_spain_trad_scores.csv (contiene todos los datos de portugal y españa con las columnas adicionales de
comentarios traducidos, scores y scores_media.
"""


import ast
import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

#USO: python predecir_algoritmo_generativo.py --csv portugal_spain_trad.csv

parser=argparse.ArgumentParser(description='prediccion generativos')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='train', help='split') #esto lo he cambiado (Aitzi dixit)
parser.add_argument('--sample', type=int, default=-1, help='sample')
parser.add_argument('--csv', type=str, help='csv a evaluar', required=True)
args=parser.parse_args()
#(Aitzi  dixit, aquí complicad el prompt lo que necesitéis para evitar la verbosity)
template = """You are an expert Airbnb review evaluator. Your task is to estimate the user's satisfaction score from 0.0 (very bad) to 9.0 (excellent) based on the comment below. Do not give any explanation or ask anything, just give the number.
Comment: {comment}
Score: {score}"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model,temperature=0, num_predict=1) #deterministic (Aitzi dixit, esto también hay que modificarlo para que no se limite a devolver solo una palabra. temperature=0 es para que sea determinista y siempre de lo mismo)
chain = prompt | model

#print(df.head())
df=pd.read_csv(args.csv) #el csv tiene la columna:comments_traducidos
scores_fila=[]
scores_columna=[]
scores_media_columna=[]

for n,fila in df.iterrows(): #por cada fila (en la que hay cero o varios comentarios)
    print("n: " + str(n))
    print(fila["_id"])
    comments=fila["comments_traducidos"]
    c = ast.literal_eval(comments)
    for comment in c: #recorremos la lista de comentarios
        if "This is an automatic message" in comment: #si es un comentario automático, lo ignoramos porque no lo ha escrito un usuario,
            continue                                  #no expresa satisfacción o insatisfacción
        ans = chain.invoke({'comment': comment, 'score': ''}).strip()  #predecimos el score de un comentario
        ans=float(ans) #pasamos el número en string que nos ha devuelto el algoritmo a int
        scores_fila.append(ans) #añadimos el score predecido a la lista de scores de una fila
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

nombre_salida_csv="g_"+os.path.splitext(args.csv)[0]+"_scores.csv" #nombre del csv en el q se van a guardar los comentarios traducidos con scores

#guardamos en csv
df.to_csv(nombre_salida_csv,index=False)

print("El csv g_"+os.path.splitext(args.csv)[0]+"_scores.csv se ha guardado correctamente.")
#g_ para indicar que se han predicho los scores con un algoritmo generativo (ollama/gemma2b)