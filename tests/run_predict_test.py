from fastapi.testclient import TestClient
from src.serve.api import app

with TestClient(app) as client:
    r = client.post('/predict', json={'store_id':2, 'start_date':'2015-08-01', 'horizon':7})
    print('STATUS', r.status_code)
    print(r.text)
