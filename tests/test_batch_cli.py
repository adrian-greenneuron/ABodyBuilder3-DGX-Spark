"""Exercise real CLI dispatch, formatting, bounded retries and explicit failures."""
import csv, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
if not Path("/opt/abodybuilder3/scripts").is_dir():
    raise unittest.SkipTest("Run these CLI integration checks inside the deployment image.")
import numpy as np
import torch
from typer.testing import CliRunner

SCRIPTS=Path('/opt/abodybuilder3/scripts')
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('cli_test',SCRIPTS/'predict.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
import abodybuilder3.batched_inference as batching

class BatchCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.csv=self.root/'input.csv';self.report=self.root/'report.json'
        with self.csv.open('w') as f:
            w=csv.writer(f);w.writerow(['name','heavy','light'])
            for i in range(5):w.writerow(['duplicate','AAAA','LLLL'])
        self.args=['predict','--csv',str(self.csv),'--output',str(self.root/'out'),
                   '--device','cpu','--workers','1','--batch-size','4','--batch-report',str(self.report),'--quiet']
        self.runner=CliRunner()
    def fake(self,model,prepared,device):
        batch,inputs=prepared;out=[]
        for inp in inputs:
            n=len(inp['aatype']);mask=np.zeros((n,37));mask[:,1]=1
            out.append({'atom37':np.zeros((n,37,3)),'atom37_atom_exists':mask})
        return out,inputs
    def invoke(self,extra=(),infer=None):
        with patch.object(cli,'load_model',return_value=object()),patch.object(batching,'infer_prepared_batch',side_effect=infer or self.fake):
            return self.runner.invoke(cli.app,self.args+list(extra))
    def test_dispatch_and_duplicate_names_and_short_final_batch(self):
        r=self.invoke();self.assertEqual(r.exit_code,0,(r.output,r.exception))
        report=json.loads(self.report.read_text())
        self.assertTrue(report['completed']);self.assertEqual([x['size'] for x in report['batches']],[4,1])
        self.assertEqual({p.stem for p in (self.root/'out').glob('*.pdb')},
                         {'duplicate','duplicate_1','duplicate_2','duplicate_3','duplicate_4'})
    def test_generated_suffix_does_not_overwrite_an_existing_input_name(self):
        with self.csv.open('w') as f:
            w=csv.writer(f);w.writerow(['name','heavy','light'])
            for name in ['a','a_1','a','a','b']:w.writerow([name,'AAAA','LLLL'])
        r=self.invoke();self.assertEqual(r.exit_code,0,(r.output,r.exception))
        self.assertEqual({p.stem for p in (self.root/'out').glob('*.pdb')},{'a','a_1','a_2','a_3','b'})
    def test_auto_uses_small_job_cap(self):
        r=self.invoke(['--batch-size','auto']);self.assertEqual(r.exit_code,0,(r.output,r.exception))
        self.assertEqual(json.loads(self.report.read_text())['max_batch_size'],16)
    def test_oom_splits_without_losing_records(self):
        def infer(model,prepared,device):
            if len(prepared[1])>2:raise torch.cuda.OutOfMemoryError('simulated')
            return self.fake(model,prepared,device)
        r=self.invoke(infer=infer);self.assertEqual(r.exit_code,0,(r.output,r.exception))
        report=json.loads(self.report.read_text());self.assertEqual(report['saved'],5)
        self.assertEqual(len(report['oom_retries']),1)
        self.assertEqual([x['size'] for x in report['batches']],[2,2,1])
    def test_other_inference_failure_is_not_retried_or_hidden(self):
        def infer(*args):raise RuntimeError('deliberate model error')
        r=self.invoke(infer=infer);self.assertNotEqual(r.exit_code,0)
        report=json.loads(self.report.read_text());self.assertFalse(report['completed'])
        self.assertEqual(report['oom_retries'],[])
    def test_all_output_formats(self):
        for fmt in ['json','cif']:
            r=self.invoke(['--format',fmt]);self.assertEqual(r.exit_code,0,(r.output,r.exception))
            paths=list((self.root/'out').glob('*.'+fmt));self.assertEqual(len(paths),5)
            if fmt=='json':self.assertEqual(json.loads(paths[0].read_text())['num_residues'],8)
    def test_invalid_or_unsupported_options_fail_before_model_load(self):
        for options in [['--batch-size','0'],['--batch-size','bad'],['--batch-memory-gib','nan'],
                        ['--model','language'],['--workers','0'],['--workers','8']]:
            with self.subTest(options=options),patch.object(cli,'load_model') as load:
                r=self.runner.invoke(cli.app,self.args+options)
                self.assertNotEqual(r.exit_code,0);load.assert_not_called()

    def test_invalid_sequence_or_name_is_rejected_before_model_load(self):
        for name, heavy, light in [('bad', 'AA*', 'LL'), ('../escape', 'AA', 'LL'),
                                    ('', 'AA', 'LL'), ('missing_chain', 'AA', '')]:
            with self.csv.open('w') as f:
                writer=csv.writer(f);writer.writerow(['name','heavy','light'])
                writer.writerow([name,heavy,light])
            with self.subTest(name=name),patch.object(cli,'load_model') as load:
                r=self.runner.invoke(cli.app,self.args)
                self.assertNotEqual(r.exit_code,0);load.assert_not_called()

    def test_existing_output_requires_explicit_overwrite(self):
        r=self.invoke();self.assertEqual(r.exit_code,0,(r.output,r.exception))
        path=self.root/'out/duplicate.pdb';path.write_text('retained output')
        r=self.invoke();self.assertNotEqual(r.exit_code,0)
        self.assertEqual(path.read_text(),'retained output')
        r=self.invoke(['--overwrite']);self.assertEqual(r.exit_code,0,(r.output,r.exception))
        self.assertNotEqual(path.read_text(),'retained output')

    def test_missing_custom_checkpoint_does_not_load_released_weights(self):
        with patch.object(cli,'load_model') as load:
            r=self.runner.invoke(cli.app,self.args+['--checkpoint',str(self.root/'absent.ckpt')])
            self.assertNotEqual(r.exit_code,0);load.assert_not_called()

    def test_rescue_policy_reaches_the_preparation_controller(self):
        for extra, expected in [([], 'local'), (['--rescue', 'restrained'], 'restrained'),
                                (['--rescue', 'off'], 'off')]:
            with self.subTest(policy=expected),patch.object(cli,'prepare_output_paths') as prepare:
                r=self.invoke(['--add-hydrogens','--overwrite']+extra)
                self.assertEqual(r.exit_code,0,(r.output,r.exception))
                prepare.assert_called_once()
                paths, options, workers=prepare.call_args.args
                self.assertEqual(len(paths),5)
                self.assertEqual(options.rescue_policy,expected)
                self.assertGreater(workers,1)

    def test_invalid_rescue_options_fail_before_model_loading(self):
        for extra in [['--rescue','local'], ['--add-hydrogens','--rescue','unknown'],
                      ['--add-hydrogens','--ph','5'],
                      ['--relax','restrained','--rescue','local'],
                      ['--add-hydrogens','--preparation-workers','21']]:
            with self.subTest(extra=extra),patch.object(cli,'load_model') as load:
                r=self.runner.invoke(cli.app,self.args+extra)
                self.assertNotEqual(r.exit_code,0)
                load.assert_not_called()

if __name__=='__main__':unittest.main()
