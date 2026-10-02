import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile


ARCHIVES = {
    'ccx_2.23.tar.bz2': (
        'https://www.dhondt.de/ccx_2.23.tar.bz2',
        '35d426fed5eb164fbbaaafa20819d13d22e30bc2b9bc9c6c4ed957e9468355dd'),
    'libgfortran-ng-7.5.0.tar.bz2': (
        'https://api.anaconda.org/download/conda-forge/libgfortran-ng/7.5.0/linux-64/libgfortran-ng-7.5.0-hdf63c60_6.tar.bz2',
        '0da447d464468be5c71e73e6f8fbbfd59cc0ceeecbf7a95c19d7460bd3cb7133'),
    'FEBio-4.13.tar.gz': (
        'https://codeload.github.com/febiosoftware/FEBio/tar.gz/refs/tags/v4.13',
        'efc08104762a05f79b25fb3038bbb0f8ef84a7ae97e897983fc5a98b42e52f31'),
    'OpenRadioss_linux64.zip': (
        'https://github.com/OpenRadioss/OpenRadioss/releases/download/latest-20260728/OpenRadioss_linux64.zip',
        '598ed7b2905a7bacc8d1781470c250ac79d7558c39ba962768edf3644644fe33'),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(name, directory):
    url, expected = ARCHIVES[name]
    path = directory / name
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {name}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        partial.replace(path)
    if sha256(path) != expected:
        raise ValueError(f'Archive checksum mismatch: {path}')
    return path


def run(command):
    print(' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('The solver bundle requires Linux x86_64')
    if args.jobs < 1 or args.threads < 1:
        parser.error('Worker counts must be positive')
    root = Path(__file__).resolve().parents[1]
    runtime = root / 'runtime'
    downloads = runtime / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    archives = {name: download(name, downloads) for name in ARCHIVES}
    ccx = runtime / 'calculix'
    ccx.mkdir(exist_ok=True)
    with tarfile.open(archives['ccx_2.23.tar.bz2']) as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith('/ccx_2.23'))
        (ccx / 'ccx_2.23').write_bytes(archive.extractfile(member).read())
    (ccx / 'ccx_2.23').chmod(0o755)
    with tarfile.open(archives['libgfortran-ng-7.5.0.tar.bz2']) as archive:
        (ccx / 'libgfortran.so.4.0.0').write_bytes(archive.extractfile('lib/libgfortran.so.4.0.0').read())
        licenses = ccx / 'licenses'
        licenses.mkdir(exist_ok=True)
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('info/licenses/'):
                (licenses / Path(member.name).name).write_bytes(archive.extractfile(member).read())
    link = ccx / 'libgfortran.so.4'
    if not link.is_symlink():
        link.symlink_to('libgfortran.so.4.0.0')
    (ccx / 'manifest.json').write_text(json.dumps({
        'version': '2.23',
        'files': {p.name: sha256(p) for p in (ccx / 'ccx_2.23', ccx / 'libgfortran.so.4.0.0')}
    }, indent=2) + '\n')
    febio = runtime / 'FEBio-4.13'
    if not febio.exists():
        with tarfile.open(archives['FEBio-4.13.tar.gz']) as archive:
            archive.extractall(runtime, filter='data')
    radioss = runtime / 'OpenRadioss'
    if not radioss.exists():
        with zipfile.ZipFile(archives['OpenRadioss_linux64.zip']) as archive:
            archive.extractall(runtime)
            for member in archive.infolist():
                mode = member.external_attr >> 16
                if mode:
                    (runtime / member.filename).chmod(mode & 0o777)
    prefix = Path(sys.prefix)
    compiler = os.environ.get('CXX', 'c++')
    build = runtime / 'febio_build'
    run(['cmake', '-S', febio, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_CXX_COMPILER=' + compiler,
         '-DCMAKE_C_COMPILER=' + os.environ.get('CC', 'cc'),
         '-DUSE_MKL=ON', '-DMKL_INC=' + str(prefix / 'include'),
         '-DMKL_LIB_DIR=' + str(prefix / 'lib'),
         '-DMKL_OMP_LIB=' + str(prefix / 'lib/libiomp5.so'),
         '-DUSE_HYPRE=OFF', '-DUSE_MMG=OFF', '-DUSE_LEVMAR=OFF',
         '-DUSE_FFTW=OFF', '-DUSE_SUPERLU_MT=OFF', '-DUSE_PDL=OFF',
         '-DUSE_NLOPT=OFF', '-DUSE_ZLIB=OFF', '-DUSE_STATIC_STDLIBS=OFF',
         '-DSET_DEVCOMMIT=OFF'])
    run(['cmake', '--build', build, '--parallel', args.jobs])
    extensions = runtime / 'solver_extensions'
    run([sys.executable, root / 'tools/build_solver_extensions.py',
         '--febio-source', febio, '--febio-lib', build / 'lib',
         '--output', extensions, '--cxx', compiler])
    config = {
        'ccx': str(ccx / 'ccx_2.23'),
        'febio': str(build / 'bin/febio4'),
        'reactive_plugin': str(extensions / 'reactive.so'),
        'radioss_root': str(radioss),
        'animation_converter': str(extensions / 'anim_to_vtk_mass'),
        'threads': args.threads,
        'timeout_s': 21600,
        'library_paths': [str(build / 'lib')],
        'febio_arguments': ['-noconfig'],
        'extra_resources': [str(prefix / 'lib/libiomp5.so')],
    }
    (runtime / 'fem.json').write_text(json.dumps(config, indent=2) + '\n')
    print('Runtime configuration: ' + str(runtime / 'fem.json'), flush=True)


if __name__ == '__main__':
    main()
