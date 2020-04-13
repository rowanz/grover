# This is just for downloading the generator. See `discrimination/` for the discrimination checkpoints

import os
import requests
import argparse

parser = argparse.ArgumentParser(description='Download a model!')
parser.add_argument(
    'model_type',
    type=str,
    help='Valid model names: (base|large|mega)',
)
model_type = parser.parse_args().model_type

model_dir = os.path.join('models', model_type)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for ext in ['data-00000-of-00001', 'index', 'meta']:
    r = requests.get(f'https://storage.googleapis.com/grover-models/{model_type}/model.ckpt.{ext}', stream=True)
    with open(os.path.join(model_dir, f'model.ckpt.{ext}'), 'wb') as f:
        file_size = int(r.headers["content-length"])
        if file_size < 1000:
            raise ValueError("File doesn't exist? idk")
        chunk_size = 1000
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    print(f"Just downloaded {model_type}/model.ckpt.{ext}!", flush=True)
