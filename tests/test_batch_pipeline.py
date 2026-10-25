"""Scheduling, boundedness and failure propagation without a model runtime."""
from pathlib import Path
import sys
import threading
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from batch_pipeline import BatchPolicy, plan_batches, run_pipeline


class BatchPipelineTests(unittest.TestCase):
    def test_mixed_lengths_preserve_identity_and_limits(self):
        items = [(str(i), 'A'*n, 'L'*5) for i,n in enumerate([10,200,20,15,250,40,10])]
        policy = BatchPolicy(3, 12000, 4)
        batches = list(plan_batches(items, policy))
        self.assertCountEqual([x[0] for b in batches for x in b], [x[0] for x in items])
        for b in batches:
            self.assertLessEqual(len(b), 3)
            self.assertEqual(len({(len(h)+len(l))//policy.length_bin for _,h,l in b}),1)
            self.assertTrue(len(b)==1 or len(b)*max(len(h)+len(l) for _,h,l in b)**2 <= 12000)

    def test_incomplete_batch_and_empty_input(self):
        items = [(str(i),'H','L') for i in range(5)]
        self.assertEqual([len(b) for b in plan_batches(items,BatchPolicy(2,100))],[2,2,1])
        self.assertEqual(list(plan_batches([],BatchPolicy())),[])

    def test_invalid_limits_and_empty_sequences(self):
        for args in [(0,1,1),(1,0,1),(1,1,0)]:
            with self.assertRaises(ValueError): BatchPolicy(*args)
        with self.assertRaises(ValueError): list(plan_batches([('a','','')],BatchPolicy()))

    def test_one_thread_owns_inference_and_every_output_written(self):
        caller = threading.get_ident()
        outputs = []; lock=threading.Lock()
        def infer(items,prepared):
            self.assertEqual(threading.get_ident(),caller)
            return [x*2 for x in prepared]
        def write(items):
            with lock: outputs.extend(items)
        run_pipeline(([i] for i in range(20)),list,infer,write,prefetch=2,writers=2)
        self.assertCountEqual(outputs,[i*2 for i in range(20)])

    def test_prefetch_does_not_consume_entire_source(self):
        consumed=[]
        def source():
            for i in range(100):
                consumed.append(i)
                yield [i]
        def infer(items,prepared):
            if items == [0]: self.assertLessEqual(len(consumed),3)
            return items
        run_pipeline(source(),list,infer,lambda x:None,prefetch=2,writers=2)

    def test_stage_failures_are_not_silently_skipped(self):
        def fail(*args): raise RuntimeError('expected failure')
        for stage in ('prepare','infer','write'):
            callbacks={'prepare':list,'infer':lambda items,prepared:prepared,'write':lambda x:None}
            callbacks[stage]=fail
            with self.subTest(stage=stage),self.assertRaisesRegex(RuntimeError,'expected failure'):
                run_pipeline([[1],[2]],**callbacks)


if __name__ == '__main__': unittest.main()
