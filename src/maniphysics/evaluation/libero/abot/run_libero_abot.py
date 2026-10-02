import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from maniphysics.evaluation.libero.smolvla.run_libero_smolvla import main
if __name__ == '__main__':
    main()
