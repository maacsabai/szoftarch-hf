from typing import Union
import logging
import time
from pdfminer.high_level import extract_text, extract_pages
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
from elasticsearch import helpers


from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# ML
from transformers import pipeline
import spacy


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

ELASTIC_INDEX_NAME = 'uploaded_data'
USER = 'elastic'
PASSWORD = '4Lnrd3BgtACXT61bP5GT'

class Document(BaseModel):
    context: str
    question: str

class Question(BaseModel):
    question: str

models_initiated = None
qa_pipeline = None
nlp = None
es_client = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/init_backend')
def init_backend():
    global  qa_pipeline, es_client, models_initiated#, nlp, ,  # storage_client, tfidf_vectorizer
    start = time.time()

    logging.info("Loading QA model...")
    qa_pipeline = pipeline("question-answering", model="mcsabai/huBert-fine-tuned-hungarian-squadv2",tokenizer="mcsabai/huBert-fine-tuned-hungarian-squadv2")#, handle_impossible_answer = True)
    logging.info("QA model loaded.")
    
    #logging.info("Loading NLP model...")
    #nlp = spacy.load("hu_core_news_lg")
    #nlp = spacy.load("hu_core_news_trf")
    #logging.info("NLP model loaded.")

   
    es_client = Elasticsearch(hosts=[{"host":"localhost", "port": 9200,"scheme":"https"}],verify_certs=False, ssl_context=False, http_auth=(USER, PASSWORD), timeout = 600)    
    
    #logging.info("Initializing Elasticsearch client...")
    #es_client = Elasticsearch(cloud_id=os.getenv('ELASTIC_CLOUD_ID'),http_auth=(os.getenv('ELASTIC_USERNAME'), os.getenv('ELASTIC_PASSWORD')))
    #logging.info("Elasticsearch client initialized.")

    models_initiated = True
    
    end = time.time()
    return { 'time': end - start}

@app.post('/predict')
def predict(document: Document):
    start = time.time()

    predictions = qa_pipeline({
    'context': document.context,
    'question': document.question
    })

    end = time.time()

    return { 'prediction': predictions, 'time':  end - start  }

@app.post('/search')
def search(question: Question):
    start = time.time()

    query = {
        "query": {
            "match": {
                "context": question.question
            }
        }
    }

    result_top1 = es_client.search(query=query['query'], index=ELASTIC_INDEX_NAME, timeout='55s', size = 1)
    
    if len(result_top1.body['hits']['hits']) == 0:
        print(result_top1.body['hits']['hits'])
        end = time.time()
        return {'qa': {'answer':'', 'start':0, 'end':0, 'score':1}, 'context': '', 'time':end-start}

    else:

        result_top1 = result_top1.body['hits']['hits'][0]

        # ******* SQUAD *******
        predictions = qa_pipeline({
            'context':  result_top1['_source']['context'],
            'question': question.question
        }, top_k=5 )#, handle_impossible_answer = True)
        predictions = predictions[0]

        end = time.time()

        return { 'prediction': predictions, 'time':  end - start  }

@app.get('/pdf_reader')
def pdf_reader():
    start = time.time()
    docs = []
    file = "data/Infojegyzet_2020_51_fintech.pdf"
    text = extract_text(file)
    text = text.replace('-\n', '').replace('\n','').replace('\r', '')
    tokenized_text = tokenize_text(text, qa_pipeline.tokenizer)
    
    #file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    #tokenized_text = tokenize_text(document.context, qa_pipeline.tokenizer)
    
    for i, offset in enumerate(tokenized_text['offset_mapping']):
        context = text[offset[0][0] : offset[-1][-1]]
        doc_to_upload = {
            
            'order_number': i,
            
            'mapping_first': offset[0][0],
            'mapping_last': offset[-1][-1],
            'context': context
        }
        docs.append(doc_to_upload)
        id_ = str(int(time.time()*100000))
        #es_client.create(index=ELASTIC_INDEX_NAME,document=doc_to_upload, id=id_)
        es_client.index(index=ELASTIC_INDEX_NAME, document=doc_to_upload, id=id_)
    end = time.time()
    return {'text': docs,  'time': end-start}

@app.get('/delete_docs')
def delete_docs():
    start = time.time()
    es_client.delete_by_query(index=ELASTIC_INDEX_NAME, body={"query": {"match_all": {}}})
    end = time.time()
    return {'text': 'DONE',  'time': end-start}

def tokenize_text(context, tokenizer, max_length = 384, stride = 128): 
    inputs = tokenizer(context, max_length=max_length, stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, add_special_tokens = False)
    return inputs

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)