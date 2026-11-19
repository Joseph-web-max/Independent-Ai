import json
from pathlib import Path
import torch
from knowledge.retriever import LocalKnowledge
from memory.store import ConversationMemory
from multimodal.image_generation.model import ToyConditionalUNet,CLASS_NAMES

ROOT=Path(__file__).resolve().parents[1]

def test_v03_manifest_and_instruction_data():
    m=json.loads((ROOT/'data/manifests/v0_3_manifest.json').read_text())
    rows=[json.loads(x) for x in (ROOT/'data/v0_3_instructions.jsonl').read_text().splitlines()]
    assert m['documents']==len(rows)>=60
    assert m['third_party_model_weights'] is False

def test_v04_v05_checkpoints_present_and_provenance():
    for v in ['0_4','0_5']:
        cp=torch.load(ROOT/f'checkpoints/independent_lm_v{v}.pt',map_location='cpu',weights_only=False)
        assert cp['training_version']==v.replace('_','.')
        assert cp['training']['third_party_pretrained_weights_used'] is False
        assert cp['training']['external_ai_apis_used'] is False
        assert len(cp['model_state_dict'])>0

def test_image_checkpoint_is_project_trained():
    cp=torch.load(ROOT/'checkpoints/image_diffusion_toy_v0_6.pt',map_location='cpu',weights_only=False)
    assert cp['metadata']['weights_randomly_initialized'] is True
    assert cp['metadata']['third_party_pretrained_weights_used'] is False
    model=ToyConditionalUNet();model.load_state_dict(cp['model_state_dict'])
    assert len(CLASS_NAMES)==6

def test_local_knowledge_retrieval():
    hit=LocalKnowledge().search('What is inflation?')
    assert hit and 'sustained increase' in hit['answer']

def test_local_memory_crud(tmp_path):
    store=ConversationMemory(tmp_path/'memory.sqlite3')
    store.add('What is inflation?','Inflation is a sustained rise in prices.')
    assert store.count()==1
    assert store.search('inflation')
    store.clear();assert store.count()==0
