import os
import tempfile
import numpy as np
from maniphysics.assessment.geometry import load_mesh

def _size_field(pts, h_fine, h_coarse, r_in, r_out, curv=0):
    import gmsh
    pf = gmsh.model.mesh.field.add('Distance')
    tags = [gmsh.model.geo.addPoint(*p, h_fine) for p in pts]
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.field.setNumbers(pf, 'PointsList', tags)
    tf = gmsh.model.mesh.field.add('Threshold')
    gmsh.model.mesh.field.setNumber(tf, 'InField', pf)
    gmsh.model.mesh.field.setNumber(tf, 'SizeMin', h_fine)
    gmsh.model.mesh.field.setNumber(tf, 'SizeMax', h_coarse)
    gmsh.model.mesh.field.setNumber(tf, 'DistMin', r_in)
    gmsh.model.mesh.field.setNumber(tf, 'DistMax', r_out)
    gmsh.model.mesh.field.setAsBackgroundMesh(tf)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', curv)
    return tf

def build(model_id, kind, contact_pts, *, h_fine, h_coarse, r_in=None, r_out=None, center=True, quad=True, straight=False, curv=0, feat_deg=40.0, source_mesh=None):
    import gmsh
    M = load_mesh(model_id) if source_mesh is None else source_mesh
    off = np.asarray(M.bounds).mean(0) if center else np.zeros(3)
    M = M.copy()
    M.vertices = np.asarray(M.vertices) - off
    r_in = 0.003 if r_in is None else r_in
    r_out = 0.01 if r_out is None else r_out
    stl = tempfile.mktemp(suffix='.stl')
    M.export(stl)
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads', 2)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.merge(stl)
        gmsh.model.mesh.classifySurfaces(feat_deg * np.pi / 180, True, True, 180 * np.pi / 180)
        gmsh.model.mesh.createGeometry()
        if kind in ('solid', 'skin_core'):
            sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in gmsh.model.getEntities(2)])
            gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()
        _size_field(np.atleast_2d(contact_pts), h_fine, h_coarse, r_in, r_out, curv)
        if kind == 'shell' and quad:
            gmsh.option.setNumber('Mesh.RecombineAll', 1)
            gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 2)
        if straight:
            gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
        gmsh.option.setNumber('Mesh.Optimize', 1)
        gmsh.model.mesh.generate(3 if kind in ('solid', 'skin_core') else 2)
        if kind == 'shell' and quad:
            gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
        gmsh.model.mesh.setOrder(2)

        def grab(dim, want):
            (et_, _, enod_) = gmsh.model.mesh.getElements(dim)
            for (i, t) in enumerate(et_):
                (name, _, _, nn, _, _) = gmsh.model.mesh.getElementProperties(t)
                if name == want:
                    return np.array(enod_[i], dtype=np.int64).reshape(-1, nn)
            raise RuntimeError(f'{want} elements were not generated')
        if kind == 'solid':
            (want, conn) = ('Tetrahedron 10', grab(3, 'Tetrahedron 10'))
            skin = None
        elif kind == 'skin_core':
            (want, conn) = ('Tetrahedron 10', grab(3, 'Tetrahedron 10'))
            skin = grab(2, 'Triangle 6')
        else:
            want = 'Quadrilateral 8' if quad else 'Triangle 6'
            (conn, skin) = (grab(2, want), None)
        (ntag, ncoord, _) = gmsh.model.mesh.getNodes()
        allc = {int(t): ncoord[3 * i:3 * i + 3] for (i, t) in enumerate(ntag)}
        used = np.unique(conn if skin is None else np.concatenate([conn.ravel(), skin.ravel()]))
        remap = {int(t): i + 1 for (i, t) in enumerate(used)}
        nodes = np.array([allc[int(t)] for t in used], dtype=float)
        elems = np.vectorize(remap.get)(conn)
        if want == 'Tetrahedron 10':
            elems = elems[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
        skin = None if skin is None else np.vectorize(remap.get)(skin)
    finally:
        gmsh.finalize()
        os.unlink(stl)
    name = {'Tetrahedron 10': 'C3D10', 'Quadrilateral 8': 'S8R', 'Triangle 6': 'S6'}[want]
    return (nodes, elems, name) if skin is None else (nodes, elems, name, skin)
