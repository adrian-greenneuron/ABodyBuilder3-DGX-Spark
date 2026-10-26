"""Deliberately interrupt only the named qualification job after its first batch."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    dest = root / 'interruption'
    dest.mkdir(exist_ok=False)
    tools = Path(__file__).resolve().parent
    name = 'abb3-aot-interruption-check-261026'
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--gpus', 'all', '--ipc=host',
           '--user', '1000:1000', '-e', 'HOME=/tmp', '-v', f'{tools}:/measurement:ro',
           '-v', f'{root}/inputs:/inputs:ro', '-v', f'{dest}:/results', args.image,
           'python', '-u', '/measurement/measure_deployment.py', '--input', '/inputs/antibodies_20000.csv',
           '--output', '/results', '--model', 'plddt', '--capture-limit', '0']
    (dest / 'command_261026.json').write_text(json.dumps(cmd, indent=2) + '\n')
    with (dest / 'inference.log').open('w') as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        signalled = False
        while process.poll() is None:
            if not signalled and '"event": "first_batch_complete"' in (dest / 'inference.log').read_text():
                subprocess.run(['docker', 'kill', '--signal=SIGINT', name], check=True, stdout=subprocess.DEVNULL)
                signalled = True
            time.sleep(.1)
        code = process.wait()
    audit = json.loads((dest / 'batching.json').read_text())
    outputs = list((dest / 'pdbs').glob('*.pdb'))
    complete = all(path.read_text().endswith('END\n') for path in outputs)
    if not signalled or code == 0 or audit['completed'] or not complete or len(outputs) != audit['saved']:
        raise ValueError(f'Interruption accounting failed: signalled={signalled}, exit={code}, files={len(outputs)}, audit={audit}')
    report = {'completed': True, 'test': 'deliberate interruption after the first successful batch',
              'signal': 'SIGINT', 'exit_code': code, 'partial_output_count': len(outputs),
              'all_retained_PDBs_complete': complete, 'failed_job_marked_incomplete': not audit['completed']}
    (root / 'interruption_check_261026.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
