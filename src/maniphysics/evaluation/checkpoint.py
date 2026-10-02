from pathlib import Path
import shutil


def isolated_checkpoint(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    destination.mkdir(parents=True, exist_ok=False)
    weights = {'.safetensors', '.bin', '.pt', '.pth'}
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if '.back.' in path.name or '__pycache__' in relative.parts:
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in weights:
                target.symlink_to(path.resolve())
            else:
                shutil.copy2(path, target)
    return str(destination)
