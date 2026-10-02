import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--febio-source', type=Path, required=True)
    parser.add_argument('--febio-lib', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plugin = root / 'src/maniphysics/assessment/solvers/plugin/reactive.cpp'
    subprocess.run([args.cxx, '-std=c++17', '-O2', '-fPIC', '-shared', '-fopenmp', '-DLINUX',
                    '-I' + str(args.febio_source.resolve()), str(plugin),
                    '-L' + str(args.febio_lib.resolve()), '-lfebiomech', '-lfecore',
                    '-o', str(output / 'reactive.so')], check=True)
    subprocess.run([args.cxx, '-std=c++17', '-O2', '-DLINUX',
                    str(root / 'third_party/radioss_output/anim_to_vtk_mass.cpp'),
                    '-o', str(output / 'anim_to_vtk_mass')], check=True)


if __name__ == '__main__':
    main()
