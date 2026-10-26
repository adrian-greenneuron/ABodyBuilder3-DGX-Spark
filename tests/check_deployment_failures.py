"""Real installed-CLI failure and optional-path checks in the qualified image."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from typer.testing import CliRunner


def main():
    out = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location('failure_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fixture = json.loads(Path('/opt/abodybuilder3/tests/benchmark_antibodies.json').read_text())[0]
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = ['predict', '--heavy', fixture['heavy'], '--light', fixture['light'],
                '--output', str(root / 'prediction.pdb'), '--seed', '42', '--quiet']
        result = runner.invoke(cli.app, base + ['--execution', 'aot', '--aot-root', str(root / 'missing')])
        if result.exit_code == 0 or list(root.glob('*.pdb')):
            raise ValueError('Missing packages silently succeeded or wrote a prediction.')
        if 'manifest missing' not in str(result.exception):
            raise ValueError(f'Unexpected failure: {result.output} {result.exception}')
        record = {'missing_package_exit_code': result.exit_code, 'missing_package_error': str(result.exception),
                  'no_predictions_on_package_failure': True}
        # Policy/interface checks here reuse the established chemistry tests;
        # the following real CLI call qualifies its CUDA AOT/raw-output boundary.
        result = runner.invoke(cli.app, base + ['--add-hydrogens', '--ph', '7'])
        if result.exit_code:
            raise ValueError(f'Optional preparation failed: {result.output} {result.exception}')
        raw = root / 'prediction.pdb'
        prepared = root / 'prediction.prepared.pdb'
        audit = json.loads((root / 'prediction.preparation.json').read_text())
        if not raw.is_file() or not prepared.is_file():
            raise ValueError('Missing separate raw/prepared outputs.')
        record['preparation_audit'] = audit
        record['raw_prediction_preserved'] = raw.read_text() != prepared.read_text()
        record['completed'] = True
    out.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
