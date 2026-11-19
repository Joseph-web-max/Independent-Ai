"""Independent AI Lab v1.0 research-preview API; local models only."""
from __future__ import annotations
import base64,io,sys
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from PIL import Image
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from ai_core.model import IndependentLanguageModel,ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer
from memory.store import ConversationMemory
from safety.guardrails import validate_prompt
from knowledge.retriever import LocalKnowledge
from multimodal.image_generation.model import ToyConditionalUNet,sample,prompt_to_class,CLASS_NAMES

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
TEXT_CHECKPOINT=ROOT/'checkpoints/independent_lm_v0_5.pt'
IMAGE_CHECKPOINT=ROOT/'checkpoints/image_diffusion_toy_v0_6.pt'
MEMORY=ConversationMemory(ROOT/'data/local_conversations.sqlite3')
KNOWLEDGE=LocalKnowledge()
app=FastAPI(title='Independent AI Lab',version='1.0.0',description='Local experimental language and synthetic-pattern generation models. No external AI APIs.')
text_model=None;image_model=None;tokenizer=ByteTokenizer()

class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=350)
    remember:bool=True
    max_new_tokens:int=Field(default=80,ge=8,le=120)
class ChatResponse(BaseModel):
    response:str
    model:str='Independent AI Lab v0.5 instruction-tuned research model'
    memory_used:int=0
    source:str='model_generation'
    match_score:float=0.0
    research_preview:bool=True
class ImageRequest(BaseModel):
    prompt:str=Field(min_length=1,max_length=240)
class ImageResponse(BaseModel):
    image_base64:str
    prompt:str
    category:str
    model:str='Independent AI Lab toy diffusion v0.6'
    limitations:str='32x32 synthetic diffusion experiment; samples remain noisy and are not general text-to-image.'

def load_text_model():
    global text_model
    if text_model is None:
        if not TEXT_CHECKPOINT.exists():raise RuntimeError('v0.5 text checkpoint missing; see checkpoints/.')
        cp=torch.load(TEXT_CHECKPOINT,map_location=DEVICE,weights_only=False)
        cfg=ModelConfig(**cp['model_config']);m=IndependentLanguageModel(cfg).to(DEVICE);m.load_state_dict(cp['model_state_dict']);m.eval();text_model=m
    return text_model

def load_image_model():
    global image_model
    if image_model is None:
        if not IMAGE_CHECKPOINT.exists():raise RuntimeError('v0.6 image checkpoint missing; train it using multimodal/image_generation/train_toy_diffusion.py.')
        cp=torch.load(IMAGE_CHECKPOINT,map_location=DEVICE,weights_only=False)
        m=ToyConditionalUNet().to(DEVICE);m.load_state_dict(cp['model_state_dict']);m.eval();image_model=m
    return image_model

@app.get('/',include_in_schema=False)
def home():return FileResponse(ROOT/'index.html')

@app.get('/api/health')
def health():
    return {'status':'ready','release':'1.0.0-research-preview','device':str(DEVICE),'external_ai_apis':False,'text_checkpoint_present':TEXT_CHECKPOINT.exists(),'image_checkpoint_present':IMAGE_CHECKPOINT.exists(),'memory_turns':MEMORY.count(),'capabilities':{'text_generation':'experimental-trained','instruction_tuning':'small-dataset-experimental','image_generation':'experimental-noisy-synthetic-diffusion','image_understanding':'not_implemented','speech_recognition':'not_trained','speech_generation':'not_trained','video_understanding':'not_trained','video_generation':'not_trained','advanced_reasoning':'not_established'}}

@app.get('/api/capabilities')
def capabilities():
    return {'implemented':['local text generation','local Q&A retrieval over project corpus','experimental instruction tuning','local SQLite conversation memory','toy synthetic diffusion sampler (outputs remain noisy)'],'not_implemented':['reliable frontier-grade reasoning','image understanding','speech recognition','speech synthesis','video understanding','video generation','web search'],'notice':'This research prototype can produce incorrect or incoherent responses.'}

@app.post('/api/chat',response_model=ChatResponse)
def chat(req:ChatRequest):
    try:
        user=validate_prompt(req.message)
        # When a close question match exists, return its authored training answer
        # rather than pretending the tiny language model can answer reliably.
        hit=KNOWLEDGE.search(user)
        if hit:
            if req.remember: MEMORY.add(user,hit['answer'])
            return ChatResponse(response=hit['answer'],source='local_knowledge_retrieval',match_score=hit['score'],memory_used=0)
        model=load_text_model();matches=MEMORY.search(user,limit=2) if req.remember else []
        memory_line=''
        if matches:
            short=' | '.join(f"Earlier you asked: {m['user'][:55]}" for m in matches)
            memory_line=('MEMORY: '+short+'\n')[:55]
        prefix=memory_line+'USER: '+user+'\nASSISTANT:'
        encoded=tokenizer.encode(prefix)
        if len(encoded)>=model.config.context_length-4:
            encoded=encoded[-(model.config.context_length-4):]
        ids=torch.tensor([encoded],dtype=torch.long,device=DEVICE)
        with torch.inference_mode():out=model.generate(ids,max_new_tokens=req.max_new_tokens,temperature=0.75,top_k=24)
        response=tokenizer.decode(out[0,len(encoded):].tolist()).split('\nUSER:',1)[0].strip()
        if not response:response='I could not form a response. This small experimental model may not understand the prompt.'
        if req.remember:MEMORY.add(user,response)
        return ChatResponse(response=response,source='small_model_generation',match_score=0.0,memory_used=len(matches))
    except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc)) from exc
    except Exception as exc:raise HTTPException(status_code=500,detail=f'Local model error: {exc}') from exc

@app.post('/api/image',response_model=ImageResponse)
def image(req:ImageRequest):
    try:
        model=load_image_model();label=prompt_to_class(req.prompt);labels=torch.tensor([label],device=DEVICE)
        tensor=sample(model,labels,steps=40,image_size=32,device=DEVICE)[0].detach().cpu()
        arr=((tensor.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')
        im=Image.fromarray(arr,'RGB').resize((256,256),Image.Resampling.NEAREST)
        buff=io.BytesIO();im.save(buff,format='PNG')
        return ImageResponse(image_base64=base64.b64encode(buff.getvalue()).decode('ascii'),prompt=req.prompt,category=CLASS_NAMES[label])
    except Exception as exc:raise HTTPException(status_code=500,detail=f'Image model error: {exc}') from exc

@app.get('/api/memory')
def memory_recent():return {'turns':MEMORY.recent(20),'count':MEMORY.count(),'storage':'local SQLite file in data/local_conversations.sqlite3'}

@app.delete('/api/memory')
def memory_clear():
    MEMORY.clear();return {'cleared':True,'count':0}
