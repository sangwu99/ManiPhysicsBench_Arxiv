from maniphysics.paths import release_root
import json
from pathlib import Path
DEFAULT_JSON = release_root() / 'bench/assets/registry.json'
DEFAULTS = dict(material_class='rigid', static_friction=0.5, dynamic_friction=0.5, restitution=0.0, density=None, standing=False, tilt_limit_deg=45.0, topple_persist_s=0.5, rot_slip_limit_deg=30.0, spill=None)

class PhysPropertyRegistry:

    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_JSON
        self.db = json.loads(self.path.read_text())

    def props(self, model_id):
        merged = dict(DEFAULTS)
        merged.update(self.db.get(model_id, {}))
        return merged

    def model_db_override(self):
        db = self.db
        out = {}
        for (mid, p) in db.items():
            entry = {}
            if p.get('density') is not None:
                entry['density'] = p['density']
            for k in ('bbox', 'scales'):
                if k in p:
                    entry[k] = p[k]
            if entry:
                out[mid] = entry
        return out
