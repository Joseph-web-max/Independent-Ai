"""Run the v0.5 model locally in the terminal (experimental generation only)."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ai_core.model import IndependentLanguageModel,ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/independent_lm_v0_5.pt'));p.add_argument('--tokens',type=int,default=80);a=p.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');cp=torch.load(a.checkpoint,map_location=device,weights_only=False)
 model=IndependentLanguageModel(ModelConfig(**cp['model_config'])).to(device);model.load_state_dict(cp['model_state_dict']);model.eval();tok=ByteTokenizer()
 print('Independent AI Lab v0.5 — experimental text generation; answers may be incoherent. Type /quit to exit.')
 while True:
  try:q=input('You: ').strip()
  except (EOFError,KeyboardInterrupt):break
  if q.lower() in {'/quit','/exit'}:break
  if not q:continue
  prompt='USER: '+q[:120]+'\nASSISTANT:';ids=torch.tensor([tok.encode(prompt)],device=device)
  with torch.inference_mode():out=model.generate(ids,max_new_tokens=a.tokens,temperature=.72,top_k=20)
  print('AI (experimental):',tok.decode(out[0,ids.shape[1]:].tolist()).split('\nUSER:',1)[0].strip(),'\n')
if __name__=='__main__':main()
