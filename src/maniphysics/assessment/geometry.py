import trimesh
from maniphysics.zoo.catalog import geometry_path


def load_mesh(object_id):
    mesh = trimesh.load(geometry_path(object_id), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh
