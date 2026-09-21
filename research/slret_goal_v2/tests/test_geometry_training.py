import unittest
import torch
from methods.seds_adaptation.geometry_training import (
    geometry_learning_rates,adaptation_state,load_adaptation,configure_geometry_fusion)
from methods.seds_adaptation.pose3d_branch import Pose3DBranch


class GeometryTrainingTests(unittest.TestCase):
    def test_frozen_fusion_allows_geometry_gradients_only(self):
        model=self.model()
        configure_geometry_fusion(model,freeze_fusion=True)
        before={k:v.clone() for k,v in model.state_dict().items()}
        self.assertFalse(model.fusion.training)
        opt=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.01)
        groups=geometry_learning_rates(opt,model,1e-4,1e-5,freeze_fusion=True)
        self.assertEqual({g['adaptation_group'] for g in groups},{'geometry'})
        x=torch.randn(2,4,1,49,3);valid=torch.ones(2,4,1,dtype=torch.bool)
        model.fusion(model.pose3d_branch(x,valid)).sum().backward()
        self.assertGreater(model.pose3d_branch.output.weight.grad.abs().sum(),0)
        self.assertIsNone(model.fusion.weight.grad)
        opt.step()
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        self.assertIn('pose3d_branch.output.weight',changed)
        self.assertTrue(all(k.startswith('pose3d_branch.') for k in changed))

    def test_matched_control_fusion_only(self):
        model=self.model()
        configure_geometry_fusion(model,fusion_only=True)
        before={k:v.clone() for k,v in model.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.01)
        groups=geometry_learning_rates(opt,model,1e-4,1e-5,fusion_only=True)
        self.assertEqual({g['adaptation_group'] for g in groups},{'fusion'})
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.ones_like(p)
        opt.step()
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith('fusion.') for k in changed))
        self.assertEqual(model.pose3d_branch.output.weight.count_nonzero(),0)

    def model(self):
        model=torch.nn.Module()
        model.original=torch.nn.Linear(4,4)
        model.fusion=torch.nn.Linear(4,4)
        model.pose3d_branch=Pose3DBranch(4,4,clip_temporal=True)
        return model

    def test_parameter_groups_and_compact_reconstruction(self):
        torch.manual_seed(42)
        model=self.model()
        base={k:v.clone() for k,v in model.state_dict().items()}
        configure_geometry_fusion(model)
        self.assertFalse(model.original.training)
        self.assertTrue(model.training)
        opt=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.01)
        groups=geometry_learning_rates(opt,model,1e-4,1e-5)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in groups},
                         {'geometry':1e-4,'fusion':1e-5})
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.ones_like(p)
        opt.step()
        delta=adaptation_state(model)
        rebuilt=self.model();rebuilt.load_state_dict(base)
        load_adaptation(rebuilt,delta)
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,rebuilt.state_dict()[key]))
        self.assertTrue(torch.equal(model.original.weight,base['original.weight']))
        with self.assertRaises(ValueError):load_adaptation(rebuilt,{})

    def test_train_input_forwarding_keeps_original(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'slret_goal/tools'))
        from seds_runtime import model_inputs
        keys=['right_pose','left_pose','body_pose','body_clips_start','body_mask',
              'RGB_feature','pairs_text','pairs_segment','pairs_mask','pairs_text_aug','pairs_mask_aug',
              'geometry_windows','geometry_valid']
        batch={k:torch.ones(1) for k in keys}
        result=model_inputs(batch,'cpu');body=result[5]
        self.assertIs(body['pose'],batch['body_pose'])
        self.assertIs(body['rgb'],batch['RGB_feature'])
        self.assertIs(body['geometry_windows'],batch['geometry_windows'])
        self.assertIs(body['geometry_valid'],batch['geometry_valid'])
