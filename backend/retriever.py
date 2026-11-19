"""Transparent lexical retrieval over the project's own Q&A documents.

This is retrieval, not a learned embedding model or proof of reasoning.
"""
from __future__ import annotations
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STOP={'what','whats','is','are','the','a','an','of','to','in','on','for','and','or','how','why','when','where','explain','define','tell','me','about','please','can','you','does','do','difference','between'}

def tokens(text:str)->set[str]:
    return {w for w in re.findall(r"[a-z0-9]+",text.lower()) if len(w)>1 and w not in STOP}

def load_qa():
    rows=[]
    for filename in ['v0_2_additions.txt','v0_3_additions.txt']:
        raw=(ROOT/'data'/filename).read_text(encoding='utf-8')
        for doc in raw.split('<|document|>'):
            q=re.search(r'USER:\s*(.*?)\s*ASSISTANT:',doc,re.S)
            a=re.search(r'ASSISTANT:\s*(.*)',doc,re.S)
            if q and a:
                question=' '.join(q.group(1).split());answer=' '.join(a.group(1).split())
                if question and answer:rows.append({'question':question,'answer':answer,'source':filename})
    return rows

class LocalKnowledge:
    def __init__(self):
        self.rows=load_qa();self.doc_tokens=[tokens(r['question']) for r in self.rows]
    def search(self,query:str):
        q=tokens(query)
        if not q:return None
        ranked=[]
        for row,dt in zip(self.rows,self.doc_tokens):
            overlap=len(q&dt)
            if not overlap:continue
            precision=overlap/max(1,len(q));recall=overlap/max(1,len(dt))
            score=2*precision*recall/max(1e-9,precision+recall)
            if dt.issubset(q):score+=0.12
            ranked.append((score,row))
        if not ranked:return None
        ranked.sort(key=lambda x:x[0],reverse=True)
        score,row=ranked[0]
        threshold=0.34 if len(q)>1 else 0.90
        if score<threshold:return None
        return {'answer':row['answer'],'matched_question':row['question'],'score':round(min(score,1.0),3),'source':row['source']}
