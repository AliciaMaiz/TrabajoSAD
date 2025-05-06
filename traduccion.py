#ollama run gemma2:2b
#ollama pull gemma2:2b
#python traduccion.py --sample 5
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.evaluation import ExactMatchStringEvaluator
from datasets import load_dataset
import argparse
parser=argparse.ArgumentParser(description='casiMedicos ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='train', help='split') #esto lo he cambiado (Aitzi dixit)
parser.add_argument('--sample', type=int, default=-1, help='sample')
args=parser.parse_args()
#(Aitzi  dixit, aquí complicad el prompt lo que necesitéis para evitar la verbosity)
template = """Supose you are a professional translator. translate into an informal English the following text. Do not translate if the text its already in English.
Text: {text}
Translation: {translation}"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model,temperature=0) #deterministic (Aitzi dixit, esto también hay que modificarlo para que no se limite a devolver solo una palabra. temperature=0 es para que sea determinista y siempre de lo mismo)
chain = prompt | model
dataset="MongoDB/airbnb_embeddings"
airBB = load_dataset(dataset) #check huggingface datasets for details
for n,instance in enumerate(airBB[args.split]):
    if n==args.sample: break #speed up things use only the first n instances
    reviews= instance['reviews']
    if not reviews:
        continue #si reviews está vacío (reviews=[]), no traducimos
    comment = reviews[0]['comments']


    ans=chain.invoke({'text': comment, 'translation': ''}).strip() #remove newLine
    print("\n"+Fore.GREEN + "| " + args.model + " | " + dataset + "-" + args.lang + "-" + args.split + " | n: " + str(n + 1) + Fore.RESET)
    print(Fore.LIGHTMAGENTA_EX+ "Comentario original: " + Fore.RESET + comment)
    print(Fore.LIGHTMAGENTA_EX+ "Traducción: " + Fore.RESET + ans)

