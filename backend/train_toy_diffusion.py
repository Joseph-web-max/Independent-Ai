"""Train the tiny class-conditional diffusion demo on procedurally generated shapes."""
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from multimodal.image_generation.model import ToyConditionalUNet,DiffusionSchedule,CLASS_NAMES

def make_dataset(n:int,seed:int=20261122,size:int=32):
    rng=random.Random(seed);images=[];labels=[]
    palettes=[((15,25,48),(245,110,100)),((20,38,34),(245,205,90)),((33,24,48),(130,210,245)),((20,30,42),(230,230,230)),((38,24,26),(245,150,90)),((20,35,40),(120,220,160))]
    for i in range(n):
        label=i%len(CLASS_NAMES);bg,fg=palettes[label]
        im=Image.new('RGB',(size,size),bg);d=ImageDraw.Draw(im)
        if label==0:
            for _ in range(rng.randint(2,5)):
                r=rng.randint(3,8);x=rng.randint(0,size-1);y=rng.randint(0,size-1);d.ellipse((x-r,y-r,x+r,y+r),fill=fg)
        elif label==1:
            for _ in range(rng.randint(2,5)):
                r=rng.randint(3,7);x=rng.randint(0,size-2*r);y=rng.randint(0,size-2*r);d.rectangle((x,y,x+2*r,y+2*r),fill=fg)
        elif label==2:
            for _ in range(rng.randint(1,3)):
                x=rng.randint(2,size-15);y=rng.randint(2,size-15);w=rng.randint(8,15);h=rng.randint(8,15)
                d.polygon([(x+w//2,y),(x,y+h),(x+w,y+h)],fill=fg)
        elif label==3:
            for _ in range(rng.randint(3,6)):
                y=rng.randint(0,size-1);d.line((0,y,size,y),fill=fg,width=rng.randint(1,3))
        elif label==4:
            r=rng.randint(5,13);cx=rng.randint(r,size-r);cy=rng.randint(r,size-r)
            for k in range(3):
                rr=max(1,r-k*3);d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=fg,width=2)
        else:
            cell=rng.choice([4,6,8])
            for y in range(0,size,cell):
                for x in range(0,size,cell):
                    if (x//cell+y//cell)%2==0:d.rectangle((x,y,x+cell-1,y+cell-1),fill=fg)
        arr=np.asarray(im,dtype=np.float32)/127.5-1.0
        images.append(torch.from_numpy(arr).permute(2,0,1));labels.append(label)
    return torch.stack(images),torch.tensor(labels,dtype=torch.long)

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=360);p.add_argument('--samples',type=int,default=768);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=20261122);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    x,y=make_dataset(a.samples,a.seed);loader=DataLoader(TensorDataset(x,y),batch_size=a.batch_size,shuffle=True,drop_last=True)
    model=ToyConditionalUNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.0001)
    sched=DiffusionSchedule(40,device);history=[];start=time.time();model.train();it=iter(loader)
    print(f'v0.6 synthetic image training: {a.samples} procedural images, {len(CLASS_NAMES)} classes, {sum(p.numel() for p in model.parameters()):,} params, device={device}',flush=True)
    for step in range(1,a.steps+1):
        try:clean,labels=next(it)
        except StopIteration:it=iter(loader);clean,labels=next(it)
        clean=clean.to(device);labels=labels.to(device);t=torch.randint(0,sched.steps,(clean.shape[0],),device=device)
        noise=torch.randn_like(clean);abar=sched.alpha_bars[t].view(-1,1,1,1)
        noisy=torch.sqrt(abar)*clean+torch.sqrt(1-abar)*noise
        opt.zero_grad(set_to_none=True);pred=model(noisy,t,labels);loss=torch.nn.functional.mse_loss(pred,noise)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%60==0 or step==a.steps:
            history.append({'step':step,'noise_mse':round(float(loss.detach()),6)})
            print(f'step {step}/{a.steps} noise_mse={float(loss.detach()):.5f}',flush=True)
    out=ROOT/'checkpoints/image_diffusion_toy_v0_6.pt';out.parent.mkdir(exist_ok=True)
    metadata={'model':'ToyConditionalUNet','classes':CLASS_NAMES,'image_size':32,'diffusion_steps':sched.steps,'training_steps':a.steps,'synthetic_training_images':a.samples,'parameters':sum(p.numel() for p in model.parameters()),'device':str(device),'weights_randomly_initialized':True,'third_party_pretrained_weights_used':False,'external_ai_apis_used':False,'dataset_description':'Procedurally generated geometric pattern images created within this project; not photographs; no external image dataset.'}
    torch.save({'model_state_dict':model.state_dict(),'metadata':metadata},out)
    (ROOT/'training/image_training_v0_6.json').write_text(json.dumps({'metadata':metadata,'history':history,'elapsed_seconds':round(time.time()-start,2)},indent=2))
    print(f'completed image experiment: {out}',flush=True)
if __name__=='__main__':main()
