"""Instruction-tune the project's own v0.4 checkpoint with answer-only loss."""
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ai_core.model import IndependentLanguageModel,ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer

def load_examples():
    rows=[]
    for line in (ROOT/'data/v0_3_instructions.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows

def pack(row,tok,ctx):
    prefix=('USER: '+row['instruction'].strip()+'\nASSISTANT:').encode('utf-8')
    answer=(' '+row['response'].strip()+'\n').encode('utf-8')
    available=max(8,ctx-len(prefix))
    if available<8: prefix=prefix[-(ctx-8):]; available=8
    answer=answer[:available]
    all_bytes=prefix+answer
    ids=torch.tensor(list(all_bytes),dtype=torch.long)
    x=ids[:-1].clone(); y=ids[1:].clone()
    # y index j predicts original byte j+1; only predict bytes after the prefix.
    answer_start=max(0,len(prefix)-1)
    y[:answer_start]=-100
    if len(y)<ctx:
        pad=ctx-len(y); x=F.pad(x,(0,pad),value=0); y=F.pad(y,(0,pad),value=-100)
    else: x=x[:ctx]; y=y[:ctx]
    return x,y

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=180);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=0.000025);p.add_argument('--seed',type=int,default=20261121);p.add_argument('--device',default='auto');a=p.parse_args()
    random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    rows=load_examples(); random.Random(a.seed).shuffle(rows); split=max(1,int(len(rows)*0.1)); valrows=rows[:split]; trainrows=rows[split:]
    modeldata=torch.load(ROOT/'checkpoints/independent_lm_v0_4.pt',map_location=dev,weights_only=False)
    config=ModelConfig(**modeldata['model_config']);model=IndependentLanguageModel(config).to(dev);model.load_state_dict(modeldata['model_state_dict'])
    tok=ByteTokenizer();tr=[pack(r,tok,config.context_length) for r in trainrows];va=[pack(r,tok,config.context_length) for r in valrows]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.03)
    def eval_loss(dataset):
        model.eval(); losses=[]
        with torch.no_grad():
            for i in range(0,len(dataset),a.batch_size):
                b=dataset[i:i+a.batch_size];x=torch.stack([z[0] for z in b]).to(dev);y=torch.stack([z[1] for z in b]).to(dev)
                logits,_=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100);losses.append(float(loss))
        model.train();return sum(losses)/max(1,len(losses))
    initial=eval_loss(va);print(f'v0.5 SFT: {len(trainrows)} train pairs, {len(valrows)} heldout, initial answer loss={initial:.4f}',flush=True)
    history=[];start=time.time();model.train()
    for step in range(1,a.steps+1):
        selected=[tr[random.randrange(len(tr))] for _ in range(a.batch_size)]
        x=torch.stack([z[0] for z in selected]).to(dev);y=torch.stack([z[1] for z in selected]).to(dev)
        opt.zero_grad(set_to_none=True);logits,_=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%30==0 or step==a.steps:
            val=eval_loss(va);history.append({'step':step,'train_batch_loss':float(loss.detach()),'validation_answer_loss':val})
            print(f'step {step}/{a.steps} batch_loss={float(loss):.4f} validation={val:.4f}',flush=True)
    final=eval_loss(va)
    metadata={'fine_tuned_from':'independent_ai_lab_v0_4.pt','objective':'supervised instruction tuning; loss on answer bytes only','steps':a.steps,'training_pairs':len(trainrows),'validation_pairs':len(valrows),'initial_validation_answer_loss':initial,'final_validation_answer_loss':final,'device':str(dev),'third_party_pretrained_weights_used':False,'external_ai_apis_used':False,'limitations':'Tiny synthetic instruction corpus; overfitting and poor generalization expected.'}
    out=ROOT/'checkpoints/independent_lm_v0_5.pt';torch.save({'model_state_dict':model.state_dict(),'model_config':model.config_dict(),'training_version':'0.5','training':metadata},out)
    (ROOT/'training/training_history_v0_5.json').write_text(json.dumps({'metadata':metadata,'history':history,'elapsed_seconds':round(time.time()-start,2)},indent=2))
    print(f'completed v0.5: validation answer loss {initial:.4f} -> {final:.4f}; {out}',flush=True)
if __name__=='__main__':main()
