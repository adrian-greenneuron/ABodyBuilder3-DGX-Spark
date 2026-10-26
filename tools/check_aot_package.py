"""Load an AOTI package in a fresh process and compare its saved eager example."""
import time
START = time.perf_counter()
import argparse
import json
from pathlib import Path

import torch
import torch._inductor.codecache  # Required by this pinned NGC loader.
from benchmark_runtime import Memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    with Memory() as memory, torch.no_grad():
        values = tuple(x.to('cuda') for x in torch.load(args.package.parent/'example_inputs.pt',weights_only=True))
        expected = torch.load(args.package.parent/'eager_output.pt',weights_only=True)
        torch.cuda.synchronize()
        ready = time.perf_counter()
        model = torch._inductor.aoti_load_package(str(args.package))
        loaded = time.perf_counter()
        actual = model(*values)
        torch.cuda.synchronize()
        first = time.perf_counter()
        again = model(*values)
        torch.cuda.synchronize()
        repeated = time.perf_counter()
        from abodybuilder3.openfold.data.data_transforms import make_atom14_masks
        exists = make_atom14_masks({'aatype':values[2]})['atom14_atom_exists'].bool()
        valid = exists & values[3].bool().unsqueeze(-1)
        delta = torch.linalg.vector_norm((actual['positions'][-1]-expected['positions'].to('cuda')[-1]).double(),dim=-1)
        valid_metrics = {'max_atom_displacement_A':float(delta[valid].max()),
                         'raw_atom_rmsd_A':float(delta[valid].square().mean().sqrt()),
                         'max_ca_displacement_A':float(delta[:,:,1][values[3].bool()].max()),
                         'raw_ca_rmsd_A':float(delta[:,:,1][values[3].bool()].square().mean().sqrt())}
        result = {}
        assert actual.keys() == expected.keys()
        for k,v in actual.items():
            cpu = v.cpu()
            assert cpu.shape == expected[k].shape
            assert torch.isfinite(cpu).all()
            result[k] = {'shape':list(cpu.shape),'eager_exact':torch.equal(cpu,expected[k]),
                         'max_abs_vs_eager':float((cpu-expected[k]).abs().max()),
                         'rms_vs_eager':float((cpu-expected[k]).square().mean().sqrt()),
                         'repeat_exact':torch.equal(v,again[k])}
        assert all(r['repeat_exact'] for r in result.values())
        torch.save({k:v.cpu() for k,v in actual.items()},args.output/'aot_output.pt')
    rec = {'package':str(args.package),'torch':torch.__version__,
           'precision':'strict_fp32','import_and_input_setup_seconds':ready-START,
           'package_load_seconds':loaded-ready,'first_forward_seconds':first-loaded,
           'second_forward_seconds':repeated-first,'outputs':result,
           'valid_atom_metrics':valid_metrics,
           'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
           'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),**memory.result()}
    (args.output/'check_261026.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)


if __name__ == '__main__':main()
