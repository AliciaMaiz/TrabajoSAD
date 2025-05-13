"""Con portugal_spain.csv, este programa traduce los comentarios que hay en cada propiedad y los guarda en una lista,
luego las listas de comentarios se convierten en una nueva columna para el csv.
-Output: portugal_spain_trad.csv (contiene todos los datos de portugal y españa con una columna adicional con los comentarios traducidos).
"""
#ollama run gemma2:2b
#ollama pull gemma2:2b

import ast
import os
import pandas as pd
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

parser=argparse.ArgumentParser(description='casiMedicos ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='train', help='split') #esto lo he cambiado (Aitzi dixit)
parser.add_argument('--sample', type=int, default=-1, help='sample')

parser.add_argument('--start', type=int, default=400, help='Fila inicial para traducir')
parser.add_argument('--end', type=int, default=556, help='Fila final para traducir (no inclusiva)')

args=parser.parse_args()
#(Aitzi  dixit, aquí complicad el prompt lo que necesitéis para evitar la verbosity)
template = """Supose you are a professional translator. Translate into an informal English the following text. Do not include emojis or explanations. Just return the translated text.
Text: {text}
Translation: {translation}"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model,temperature=0) #deterministic (Aitzi dixit, esto también hay que modificarlo para que no se limite a devolver solo una palabra. temperature=0 es para que sea determinista y siempre de lo mismo)
chain = prompt | model

nombre_csv="portugal.csv" #csv a traducir

traducciones=[]
df=pd.read_csv(nombre_csv)
comentarios=[]
comentarios_columna=[]

for n,fila in df.iterrows():
    if n < args.start or n >= args.end:
        continue #salta las filas que no están en el rango

    print("n: "+str(n))
    print(fila["_id"])
    reviews=fila["reviews"]
    r = ast.literal_eval(reviews)
    for review in r:
        if "comments" in review:
            #print(review["comments"])
            ans = chain.invoke({'text': review["comments"], 'translation': ''}).strip()  # remove newLine
            comentarios.append(ans)
    #print(comentarios)
    comentarios_columna.append(comentarios)
    comentarios=[]

df_traducido=df.iloc[args.start:args.end].copy()
df_traducido["comments_traducidos"]=comentarios_columna

nombre_salida_csv=f"{os.path.splitext(nombre_csv)[0]}_trad_{args.start}_{args.end-1}.csv" #nombre del csv en el q se van a guardar los comentarios traducidos

#guardamos en csv
df_traducido.to_csv(nombre_salida_csv,index=False)


#df["comments_trad"]=comentarios_columna #añadimos la lista de comentarios final del df
#df.to_csv(nombre_salida_csv,index=False) #guardamos el csv que contiene todas las columnas + la columna de comentarios (de portugal y españa)

#df_comentarios_columna = pd.DataFrame({"comments_trad":comentarios_columna}) #esto es para guardar los comentarios traducidos en un csv
#df_comentarios_columna.to_csv(nombre_salida_csv,index=False)

"""ans=chain.invoke({'text': comment, 'translation': ''}).strip() #remove newLine
print("\n"+Fore.GREEN + "| " + args.model + " | " + dataset + "-" + args.lang + "-" + args.split + " | n: " + str(n + 1) + Fore.RESET)
print(Fore.LIGHTMAGENTA_EX+ "Comentario original: " + Fore.RESET + comment)
print(Fore.LIGHTMAGENTA_EX+ "Traducción: " + Fore.RESET + ans)"""

