import json
import os
from functools import lru_cache
from pathlib import Path

def root():
    return Path(os.environ.get('MPB_ZOO', Path(__file__).resolve().parents[3] / 'zoo/benchmark')).resolve()

@lru_cache(maxsize=None)
def catalog(directory):
    return json.loads((Path(directory) / 'catalog.json').read_text())

def object_id(name):
    for entry in catalog(str(root()))['objects']:
        if name == entry['object_id'] or name in entry['aliases']:
            return entry['object_id']
    raise KeyError(f'Unknown object: {name}')

def card(name):
    ident = object_id(name)
    entry = next((x for x in catalog(str(root()))['objects'] if x['object_id'] == ident))
    return json.loads((root() / entry['card']).read_text())

def geometry_path(name):
    return root() / catalog(str(root()))['geometry_paths'][name]
