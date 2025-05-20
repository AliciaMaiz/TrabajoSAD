
import ast
import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

parser=argparse.ArgumentParser(description='prediccion ollama LLM')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='train', help='split') #esto lo he cambiado (Aitzi dixit)
parser.add_argument('--sample', type=int, default=-1, help='sample')
args=parser.parse_args()
#(Aitzi  dixit, aquí complicad el prompt lo que necesitéis para evitar la verbosity)
template = """You are an expert Airbnb review evaluator. Your task is to estimate the user's satisfaction score from 0.0 (very bad) to 9.0 (excellent) based on the comment below. Do not give any explanation or ask anything, just give the number.

Examples:
Comment: Highly recommend, beautiful place to stay was very clean. Hannah was brilliant I told her it was my partners birthday, she had put balloons and a banner up. Definitely be back again.
Score: 9.0

Comment: Clean and comfortable accommodation, in a quiet neighborhood. Gabby is very responsive and welcoming. Everything was perfect except for a small problem with the bedroom window that didn't close completely, but the repair is planned after our departure.
Score: 8.0

Comment: Everything is very small, you can't open the window, it's claustrophobic. No breakfast. Just one coffee for two days. The cleaning ladies are very noisy in the morning. Great location and cleanliness, but very expensive.
Score: 4.0

Comment: Very small space for two people, very uncomfortable bed right up against the wall, limited kitchenware, and a mini-fridge. The room is for students, not for a vacation. For the price, it's better to look for a hotel. I WILL NOT RECOMMEND IT.
Score: 1.0

Comment: The host was very nice and responded quickly. Easy to get in. However, the apartment is very cold in the evening and at night. Also, the host did not understand why we opened the window. Because there was a strong smell in the house and the room and after using the toilet, we usually ventilate. The bathroom is very limescale and the bed is uncomfortable and the bed linen was not fresh. Regardless, 1-2 nights are bearable.
Score: 3.0

Comment: Nice room, small suite but fine. Only downside was the blind for the rooflight wouldn't stay in place, so was quite bright all night. Also quite a busy road, but it is London. Overall, very satisfied.
Score: 7.0

Comment: {comment}
Score: {score}"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model,temperature=0, num_predict=1) #deterministic (Aitzi dixit, esto también hay que modificarlo para que no se limite a devolver solo una palabra. temperature=0 es para que sea determinista y siempre de lo mismo)
chain = prompt | model

nombre_csv="spain_trad.csv" #csv a predecir (el csv tiene la columna:comments_traducidos)
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
    c = ast.literal_eval(comments)
    for comment in c: #recorremos la lista de comentarios
        #print(comment)
        ans = chain.invoke({'comment': comment, 'score': ''}).strip()  #predecimos el score de un comentario
        ans=float(ans) #pasamos el número en string que nos ha devuelto el algoritmo a int
        scores_fila.append(ans) #añadimos el score predecido a la lista de scores de una fila
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

nombre_salida_csv="ALICIA_FEW_"+os.path.splitext(nombre_csv)[0]+"_scores.csv" #nombre del csv en el q se van a guardar los comentarios traducidos con scores

#guardamos en csv
df.to_csv(nombre_salida_csv,index=False)

print("El csv "+os.path.splitext(nombre_csv)[0]+"_scores.csv se ha guardado correctamente.")