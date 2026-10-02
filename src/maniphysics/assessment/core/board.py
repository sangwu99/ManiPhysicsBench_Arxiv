import numpy as np
BOARD = dict(E1_Pa=3420000000.0, E2_Pa=1755000000.0, G12_Pa=795000000.0, nu12=0.4, thickness_m=0.00038, H0_Pa=[18550000.0, 6770000.0, 10210000.0, 18550000.0, 6770000.0, 10210000.0], power=4)

def board_ratio(stress):
    s = np.asarray(stress)
    v = BOARD['nu12']
    w = v * BOARD['E2_Pa'] / BOARD['E1_Pa']
    q = np.stack(((s[..., 0] - v * s[..., 1]) / np.sqrt(1 + v * v), (s[..., 1] - w * s[..., 0]) / np.sqrt(1 + w * w), np.sqrt(2) * s[..., 2], -s[..., 0], -s[..., 1], -np.sqrt(2) * s[..., 2]), axis=-1)
    return np.sum((np.maximum(q, 0) / np.array(BOARD['H0_Pa'])) ** BOARD['power'], axis=-1) ** (1 / BOARD['power'])
