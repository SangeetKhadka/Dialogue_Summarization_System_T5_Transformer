from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize Fastapi app
app = FastAPI(
    title='Dialogue Summarizer App',
    description='Summarization using T5',
    version='1.0'
    )

model = T5ForConditionalGeneration.from_pretrained('./model')
tokenizer = T5Tokenizer.from_pretrained('./model')

# Templating
templates = Jinja2Templates(directory='.')

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r'r/n/',' ',text)
    text = re.sub(r's+',' ',text)
    text = re.sub(r'<.*?>',' ',text)

    return text


# API Endpoints

@app.post('/summarize/')
async def create_item(dialogue: DialogueInput):
    summary = summarize_dialogue(dialogue.dialogue)
    return {'summary':summary}

@app.get('/',response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html',{'request':request})

