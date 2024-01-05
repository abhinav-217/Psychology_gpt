from django.shortcuts import render,HttpResponse
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os
import time
import random
from gtts import gTTS 
from serpapi import GoogleSearch
# DB_FAISS_PATH = '../vectorstore/db_faiss'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PATH = os.path.join(BASE_DIR, 'static')
DB_FAISS_PATH = os.path.join(BASE_DIR, 'vectorstore', 'db_faiss')
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    print("Fourth function")
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    print("Fifth function")
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    print("Third function")
    # Load the locally downloaded model here
    model_path = os.path.join(BASE_DIR, 'llama-2-7b-chat.ggmlv3.q8_0.bin')
    llm = CTransformers(
        model = model_path,
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    print("Second function")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    print(llm)
    print(qa_prompt)
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    print("First function")
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


def index(request):
    return render(request,'index.html')

def make_file_name():
    rand = random.randint(1,100000)
    ts = time.time()
    return str(rand) + str(ts)

def generate(request):
    prompt = request.POST.get('prompt') 
    print("Query is:- ")
    print(prompt)
    if(prompt is None):
        return render(request,'generate.html',{'result':'No Proper prompt found','query':'Wrong Query'})
    elif(prompt == 'no'):
        language = 'en'
        myobj = gTTS(text='This is for the testing purpose for the voicegpt', lang=language, slow=False)
        file_name = make_file_name()
        myobj.save(AUDIO_PATH+"/"+file_name+".mp3") 
        audio_file_path = "../static/"+file_name+".mp3" 
        return render(request,'generate.html',{'result':'Not found result for a no','query':'User Query','file':audio_file_path})
    else:        
        if(len(prompt)>0):
            params = {
            "engine": "google",
            "q": "Coffee",
            "api_key": "a311fa34a2134b1cbd930f709a3ad530828ac70a569e30d1c1fd8017a579b682"
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results["organic_results"]
            print(organic_results[0]['title'])
            print(organic_results[0]['link'])
            print(organic_results[1]['title'])
            print(organic_results[1]['link'])
            print(organic_results[2]['title'])
            print(organic_results[2]['link'])
            data = final_result(prompt)
            print("Query:", data['query'])
            print("Result:", data['result'])
            print("Sourced documents:", data['source_documents'])
            print("Size of the list",len(data['source_documents']))
            print(data['source_documents'][1])
            myobj = gTTS(text=data['result'], lang='en', slow=False) 
            file_name = make_file_name()
            myobj.save(AUDIO_PATH+"/"+file_name+".mp3")
            audio_file_path = "../static/"+file_name+".mp3" 
            return render(request,'generate.html',{'result':data['result'],'query':data['query'],'sources':data['source_documents'],'file':audio_file_path})
        else:
            return render(request,'generate.html',{'result':'No Proper prompt found'})
            