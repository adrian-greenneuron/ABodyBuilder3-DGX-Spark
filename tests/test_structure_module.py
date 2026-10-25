"""Final-only inference must preserve updates and the reference default API."""
import copy
import importlib.util
from pathlib import Path
import unittest
import torch
from abodybuilder3.batched_inference import prepare_cpu_batch

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class StructureModuleTests(unittest.TestCase):
    def model(self):
        module=load('tested_candidate',ROOT/'candidate/structure_module.py')
        torch.manual_seed(123)
        model=module.StructureModule(c_s=23,embed_dim=16,c_z=132,c_ipa=4,c_resnet=16,
            no_heads_ipa=2,no_qk_points=2,no_v_points=2,dropout_rate=.1,no_blocks=3,
            no_transition_layers=1,no_resnet_blocks=2,no_angles=7,trans_scale_factor=1,
            epsilon=1e-7,inf=1e7,use_original_sm=True,rotation_propagation=True,use_plddt=True)
        with torch.no_grad():
            for p in model.parameters():p.normal_(0,.1)
        return model

    def test_all_updates_retained_but_only_final_sidechain_head_runs(self):
        model=self.model().eval()
        batch,_=prepare_cpu_batch([('ACD','LG')])
        inp={'single':batch['single'],'pair':batch['pair']}
        with torch.no_grad():reference=model(inp,batch['aatype'],batch['seq_mask'])
        updates=[];angles=[];handles=[]
        for i,m in enumerate(model.bb_update_layers):
            handles.append(m.register_forward_hook(lambda m,a,o,i=i:updates.append(i)))
        for i,m in enumerate(model.angle_resnet_layers):
            handles.append(m.register_forward_hook(lambda m,a,o,i=i:angles.append(i)))
        with torch.no_grad():actual=model(inp,batch['aatype'],batch['seq_mask'],
                                         _inference_final_only=True,_inference_pair_buffer=True)
        for h in handles:h.remove()
        self.assertEqual(updates,[0,1,2]);self.assertEqual(angles,[2])
        self.assertEqual(actual['positions'].shape[0],1)
        for key in ('positions','frames','sidechain_frames','angles','unnormalized_angles','states'):
            self.assertTrue(torch.equal(reference[key][-1],actual[key][-1]),key)
        for key in ('single','plddt'):self.assertTrue(torch.equal(reference[key],actual[key]),key)

    def test_training_and_gradients_cannot_use_inference_shortcuts(self):
        model=self.model();batch,_=prepare_cpu_batch([('ACD','LG')])
        inp={'single':batch['single'],'pair':batch['pair']}
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'eval'):
            model(inp,batch['aatype'],_inference_final_only=True)
        model.eval()
        with self.assertRaisesRegex(ValueError,'gradients disabled'):
            model(inp,batch['aatype'],_inference_final_only=True)
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'offload'):
            model(inp,batch['aatype'],_inference_pair_buffer=True,_offload_inference=True)


if __name__=='__main__':unittest.main()
