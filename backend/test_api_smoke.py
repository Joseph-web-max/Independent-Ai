import base64,io
from fastapi.testclient import TestClient
from PIL import Image
from main import app

client=TestClient(app)

def test_health_ready_and_reports_limitations():
    response=client.get('/api/health')
    assert response.status_code==200
    data=response.json()
    assert data['status']=='ready'
    assert data['external_ai_apis'] is False
    assert data['capabilities']['speech_generation']=='not_trained'

def test_chat_uses_local_project_knowledge_for_covered_question():
    response=client.post('/api/chat',json={'message':'What is inflation?','remember':False})
    assert response.status_code==200
    data=response.json()
    assert data['source']=='local_knowledge_retrieval'
    assert 'purchasing power' in data['response'].lower()

def test_image_endpoint_returns_png_payload():
    response=client.post('/api/image',json={'prompt':'circles'})
    assert response.status_code==200
    raw=base64.b64decode(response.json()['image_base64'])
    image=Image.open(io.BytesIO(raw))
    assert image.format=='PNG'
    assert image.size==(256,256)
