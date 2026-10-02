import shutil
from pathlib import Path

def split_obj_parts(src, out_dir):
    txt = Path(src).read_text().splitlines()
    (verts, groups, cur) = ([], [], None)
    for ln in txt:
        if ln.startswith('v '):
            verts.append(ln)
        elif ln[:2] in ('o ', 'g '):
            cur = []
            groups.append(cur)
        elif ln.startswith('f '):
            if cur is None:
                cur = []
                groups.append(cur)
            cur.append(ln)
    groups = [g for g in groups if g]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(groups) <= 1:
        shutil.copy(src, out_dir / 'part000.obj')
        return ['part000.obj']
    names = []
    for (i, faces) in enumerate(groups):
        (used, remap, out_f) = ([], {}, [])
        for ln in faces:
            toks = ln.split()[1:]
            idx = []
            for t in toks:
                gi = int(t.split('/')[0])
                gi = gi - 1 if gi > 0 else len(verts) + gi
                if gi not in remap:
                    remap[gi] = len(used) + 1
                    used.append(verts[gi])
                idx.append(str(remap[gi]))
            out_f.append('f ' + ' '.join(idx))
        if not out_f:
            continue
        nm = f'part{i:03d}.obj'
        (out_dir / nm).write_text('\n'.join(used + out_f) + '\n')
        names.append(nm)
    return names
