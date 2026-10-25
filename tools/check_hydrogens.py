"""Qualify the two existing hydrogen-addition options without model changes."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

spec = importlib.util.spec_from_file_location('abb3_cli', '/frozen/predict_main.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
source = Path(sys.argv[1]).read_text()
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=False)


def atoms(text):
    return [x for x in text.splitlines() if x.startswith(('ATOM  ', 'HETATM'))]


reference = [x for x in atoms(source) if x[12:16].strip() == 'CA']
rows = []
for method in ['reduce', 'pdbfixer']:
    random.seed(42)
    np.random.seed(42)
    start = time.perf_counter()
    output = cli.add_hydrogens(source, method)
    elapsed = time.perf_counter() - start
    lines = atoms(output)
    hydrogens = [x for x in lines if x[76:78].strip() == 'H']
    ca = [x for x in lines if x[12:16].strip() == 'CA']
    if len(ca) != len(reference) or not hydrogens:
        raise ValueError(f'Hydrogen addition failed: {method}')
    coords = np.array([[float(x[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for x in lines])
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite coordinates')
    (dest / f'{method}.pdb').write_text(output)
    rows.append({'method': method, 'seconds': elapsed, 'residues': len(ca),
                 'hydrogen_atoms': len(hydrogens), 'atoms': len(lines), 'passed': True})
(dest / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
