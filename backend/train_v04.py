"""Continue this project's own v0.2 language model on the v0.3 corpus.

This is a small research run, not frontier pretraining. No external model/API is used.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer
from training.data import load_document_splits


def load_splits(seed:int=20261120, fraction:float=0.10):
    # Preserve the exact v0.2 document holdout: none of those held-out examples
    # are allowed back into v0.4 training. Split only the newly added v0.3 docs.
    old_train, old_val = load_document_splits(seed=20261119)
    raw=(ROOT/'data/v0_3_additions.txt').read_text(encoding='utf-8')
    docs=[p.strip() for p in raw.split('<|document|>') if 'USER:' in p and 'ASSISTANT:' in p]
    rng=random.Random(seed); idx=list(range(len(docs))); rng.shuffle(idx)
    n=max(2,int(round(len(idx)*fraction))); valset=set(idx[:n])
    train=[d for i,d in enumerate(docs) if i not in valset]
    val=[d for i,d in enumerate(docs) if i in valset]
    return old_train+'\n\n'+'\n\n'.join(train), old_val+'\n\n'+'\n\n'.join(val),len(docs),len(val)


def batchify(data,batch,ctx,device):
    if data.numel() < ctx+2: raise ValueError('Training corpus is too small for context length')
    ix=torch.randint(data.numel()-ctx-1,(batch,))
    x=torch.stack([data[i:i+ctx] for i in ix]).to(device)
    y=torch.stack([data[i+1:i+ctx+1] for i in ix]).to(device)
    return x,y

@torch.no_grad()
def estimate(model,train,val,batch,ctx,device,n=10):
    model.eval(); out={}
    for name,data in [('train',train),('validation',val)]:
        vals=[]
        for _ in range(n):
            x,y=batchify(data,batch,ctx,device); _,loss=model(x,y); vals.append(float(loss))
        out[name]=round(sum(vals)/len(vals),5)
    model.train(); return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=400); p.add_argument('--run-steps',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--learning-rate',type=float,default=0.00008)
    p.add_argument('--log-every',type=int,default=100); p.add_argument('--device',default='auto')
    p.add_argument('--resume-from',default=''); p.add_argument('--seed',type=int,default=20261120)
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.log_every)<1: p.error('steps, batch-size and log-every must be positive')
    random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    tr,va,docs,valdocs=load_splits(a.seed)
    tok=ByteTokenizer(); train=torch.tensor(tok.encode(tr),dtype=torch.long); valid=torch.tensor(tok.encode(va),dtype=torch.long)
    base_path=ROOT/'checkpoints/independent_lm_v0_2.pt'
    cp_path=ROOT/'checkpoints/independent_lm_v0_4.pt'
    model_cp=torch.load(a.resume_from or base_path,map_location=device,weights_only=False)
    config=ModelConfig(**model_cp['model_config']); model=IndependentLanguageModel(config).to(device)
    model.load_state_dict(model_cp['model_state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=0.06,betas=(0.9,0.95))
    step0=0
    if a.resume_from and 'optimizer_state_dict' in model_cp:
        opt.load_state_dict(model_cp['optimizer_state_dict']); step0=int(model_cp.get('step',0))
    run_end=a.steps if a.run_steps<=0 else min(a.steps,step0+a.run_steps)
    if step0>=a.steps: raise SystemExit('Checkpoint already reached the requested step count')
    history_path=ROOT/'training/training_history_v0_4.json'
    previous={}
    if a.resume_from and history_path.exists():
        try: previous=json.loads(history_path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError): previous={}
    history=list(previous.get('history', []))
    best=float(model_cp.get('training',{}).get('best_sampled_validation_loss', float('inf'))) if a.resume_from else float('inf')
    start=time.time()
    observed=estimate(model,train,valid,a.batch_size,config.context_length,device,8)
    initial=previous.get('metadata',{}).get('initial_checkpoint_loss',observed)
    print(f'v0.4: continuing own v0.2 checkpoint; docs={docs}, heldout={valdocs}, train_bytes={train.numel()}, val_bytes={valid.numel()}, params={model.parameter_count():,}, device={device}',flush=True)
    print(f'initial={initial}',flush=True)
    for step in range(step0+1,run_end+1):
        lr=a.learning_rate*(0.15+0.85*0.5*(1+math.cos(math.pi*step/a.steps)))
        for g in opt.param_groups:g['lr']=lr
        x,y=batchify(train,a.batch_size,config.context_length,device)
        opt.zero_grad(set_to_none=True); _,loss=model(x,y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%a.log_every==0 or step==run_end:
            metrics=estimate(model,train,valid,a.batch_size,config.context_length,device,8)
            history.append({'step':step,**metrics,'lr':lr})
            print(f"step {step}/{a.steps} train={metrics['train']:.4f} validation={metrics['validation']:.4f}",flush=True)
            if metrics['validation']<best:
                best=metrics['validation']
                torch.save({'model_state_dict':model.state_dict(),'model_config':model.config_dict(),'step':step,'training_version':'0.4','training':{'continued_from':'v0.2 own checkpoint','third_party_pretrained_weights_used':False,'external_ai_apis_used':False,'steps_completed':step,'best_sampled_validation_loss':best}},ROOT/'checkpoints/independent_lm_v0_4_best.pt')
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'model_config':model.config_dict(),'step':step,'training_version':'0.4','training':{'continued_from':'v0.2 own checkpoint','third_party_pretrained_weights_used':False,'external_ai_apis_used':False,'steps_completed':step,'best_sampled_validation_loss':best}},ROOT/'checkpoints/independent_lm_v0_4_latest.pt')
            history_path.write_text(json.dumps({'metadata':{'initial_checkpoint_loss':initial,'target_steps':a.steps},'history':history},indent=2),encoding='utf-8')
    if run_end<a.steps:
        print(f'Saved at step {run_end}; resume with --resume-from checkpoints/independent_lm_v0_4_latest.pt',flush=True); return
    final=estimate(model,train,valid,a.batch_size,config.context_length,device,20)
    metadata={'initial_checkpoint_loss':initial,'continued_from':'v0.2 own trained weights','steps':a.steps,'device':str(device),'parameter_count':model.parameter_count(),'train_documents':docs-valdocs,'heldout_documents':valdocs,'training_bytes':train.numel(),'validation_bytes':valid.numel(),'initial_loss':initial,'final_sampled_loss':final,'best_sampled_validation_loss':best,'third_party_pretrained_weights_used':False,'external_ai_apis_used':False,'corpus':['data/seed_corpus.txt','data/v0_2_additions.txt','data/v0_3_additions.txt']}
    torch.save({'model_state_dict':model.state_dict(),'model_config':model.config_dict(),'training_version':'0.4','training':metadata},cp_path)
    history_path.write_text(json.dumps({'metadata':metadata,'history':history},indent=2))
    print(f'completed v0.4: final={final} checkpoint={cp_path}',flush=True)
if __name__=='__main__': main()
