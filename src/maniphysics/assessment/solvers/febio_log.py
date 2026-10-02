import numpy as np

def read_log(path):
    (out, t, rows) = ([], None, None)
    for line in open(path):
        if line.startswith('*Step'):
            if rows:
                out.append((t, np.array(rows, float)))
            (rows, t) = ([], None)
            continue
        if line.startswith('*Time'):
            t = float(line.split('=')[1])
            continue
        if line.startswith('*'):
            continue
        p = line.split()
        if len(p) >= 2 and rows is not None:
            rows.append([float(x) for x in p])
    if rows:
        out.append((t, np.array(rows, float)))
    return out
