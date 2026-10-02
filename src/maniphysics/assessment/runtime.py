import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from .cache import file_hash


@dataclass
class Runtime:
    ccx: str | None = None
    febio: str | None = None
    reactive_plugin: str | None = None
    radioss_root: str | None = None
    animation_converter: str | None = None
    threads: int = 2
    timeout_s: float = 7200.
    library_paths: tuple = ()
    febio_arguments: tuple = ()
    extra_resources: tuple = ()
    environment_variables: dict | None = None

    def files(self, backend):
        if backend == 'calculix':
            return {'solver': self.executable(self.ccx or 'ccx')}
        if backend == 'febio':
            return {'solver': self.executable(self.febio or 'febio4'),
                    'plugin': self.executable(self.reactive_plugin, executable=False)}
        root = Path(self.radioss_root or os.environ['OPENRADIOSS_PATH'])
        return {'starter': self.executable(root / 'exec/starter_linux64_gf'),
                'engine': self.executable(root / 'exec/engine_linux64_gf'),
                'converter': self.executable(self.animation_converter),
                'standard_animation_converter': self.executable(root / 'exec/anim_to_vtk_linux64_gf'),
                'time_history_converter': self.executable(root / 'exec/th_to_csv_linux64_gf')}

    @staticmethod
    def executable(value, executable=True):
        if value is None:
            raise ValueError('Required solver or plugin path was not provided')
        path = Path(shutil.which(str(value)) or str(value)).resolve()
        if not path.is_file() or executable and not os.access(path, os.X_OK):
            raise FileNotFoundError(f'Unavailable solver resource: {value}')
        return path

    def fingerprint(self, backend):
        result = {key: file_hash(path) for key, path in self.files(backend).items()}
        for i, directory in enumerate(self.library_paths):
            for path in sorted(Path(directory).glob('*.so*')):
                if path.is_file():
                    result[f'library_{i}/{path.name}'] = file_hash(path)
        result['febio_arguments'] = list(self.febio_arguments) if backend == 'febio' else []
        result['extra_resources'] = [file_hash(self.executable(p, executable=False)) for p in self.extra_resources]
        result['threads'] = self.threads
        result['timeout_s'] = self.timeout_s
        result['environment'] = self.environment_variables or {}
        return result

    def environment(self):
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads),
                   OPENBLAS_NUM_THREADS='1')
        env.update(self.environment_variables or {})
        if self.library_paths:
            env['LD_LIBRARY_PATH'] = ':'.join(map(str, self.library_paths)) + ':' + env.get('LD_LIBRARY_PATH', '')
        return env

    def bind(self, backend):
        files = self.files(backend)
        if backend == 'calculix':
            from .core import run
            run.CCX = str(files['solver'])
            run.OFFICIAL_CCX = files['solver'].parent
            run.EXTRA_ENVIRONMENT = self.environment()
        elif backend == 'febio':
            from .solvers import febio_deck, febio_native
            febio_deck.PLUGIN = files['plugin']
            febio_native.PLUGIN = files['plugin']
        else:
            from .core import radioss
            from .solvers import radioss_audit
            radioss.ROOT = str(Path(self.radioss_root or os.environ['OPENRADIOSS_PATH']).resolve())
            radioss.EXEC = str(Path(radioss.ROOT) / 'exec')
            radioss_audit.ANIMATION_CONVERTER = files['converter']
        return files
