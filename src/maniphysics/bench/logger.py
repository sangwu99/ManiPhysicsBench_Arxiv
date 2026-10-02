import json
import numpy as np
from pathlib import Path

class TrajBuffer:

    def __init__(self):
        self.rows = []

    def append(self, **kw):
        self.rows.append(kw)

    def __len__(self):
        return len(self.rows)

    def dump(self, path, meta=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        if self.rows:
            for k in self.rows[0].keys():
                arrays[k] = np.asarray([r[k] for r in self.rows])
        np.savez_compressed(path, **arrays)
        if meta is not None:
            side = path.with_suffix('.json')
            side.write_text(json.dumps(meta, indent=1, default=str))
        return path
