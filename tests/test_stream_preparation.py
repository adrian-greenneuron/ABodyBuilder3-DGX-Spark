from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@dataclass
class Options:
    relax:str="none"


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        shutil.copy2(ROOT/"candidate/stream_preparation.py",self.root/"stream_preparation.py")
        (self.root/"preparation_pool.py").write_text('''import time
def _initialise():pass
def _prepare_one(job):
    path,options=job;started=time.monotonic();time.sleep(.03)
    return {"path":path,"status":"failed" if path.endswith("bad.pdb") else "completed","warnings":[],"started_monotonic":started,"finished_monotonic":time.monotonic()}
''')
        spec=importlib.util.spec_from_file_location("stream_case",self.root/"stream_preparation.py")
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def tearDown(self):self.directory.cleanup()

    def test_bounded_complete_results(self):
        stream=self.module.PreparationStream(Options(),4)
        for i in range(30):stream.submit(self.root/f"{i}.pdb")
        result=stream.finish()
        self.assertEqual(result["accepted"],30);self.assertEqual(result["completed"],30)
        self.assertLessEqual(result["max_inflight"],8)
        self.assertEqual(len({row["path"] for row in result["results"]}),30)
        self.assertTrue(stream.closed)

    def test_duplicate_rejected(self):
        stream=self.module.PreparationStream(Options(),1)
        stream.submit(self.root/"same.pdb")
        with self.assertRaisesRegex(ValueError,"Duplicate"):stream.submit(self.root/"same.pdb")
        self.assertEqual(stream.finish()["completed"],1)

    def test_worker_failure_propagates_after_independent_jobs_finish(self):
        stream=self.module.PreparationStream(Options(),2)
        for name in ["good.pdb","bad.pdb","other.pdb"]:stream.submit(self.root/name)
        with self.assertRaisesRegex(RuntimeError,"Preparation failed"):stream.finish()
        self.assertEqual(len(stream.results),3)
        self.assertEqual(sum(row["status"]=="completed" for row in stream.results),2)

    def test_relaxation_rejected(self):
        with self.assertRaises(ValueError):self.module.PreparationStream(Options("restrained"),2)


if __name__=="__main__":unittest.main()
