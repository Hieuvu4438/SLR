import unittest
import torch
from methods.seds_adaptation.signrep_transfer import relational_clip_loss, SignRepTransfer


class RelationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.x = torch.randn(3,6,8)
        self.valid = torch.tensor([[1,1,1,1,1,1],[1,1,1,0,0,0],[1,0,0,0,0,0]],dtype=torch.bool)

    def test_similarity_transform_and_padding(self):
        q,_ = torch.linalg.qr(torch.randn(8,8))
        y = 3*self.x@q + 5
        y[~self.valid] = 10000
        self.assertLess(relational_clip_loss(y,self.x,self.valid).item(),1e-10)

    def test_pair_structure_is_not_pointwise_or_order_loss(self):
        valid = torch.ones(3,6,dtype=torch.bool)
        changed = self.x.flip(1)
        self.assertGreater(relational_clip_loss(changed,self.x,valid).item(),1e-4)
        first = relational_clip_loss(changed,self.x,valid)
        second = relational_clip_loss(changed.flip(1),self.x.flip(1),valid)
        torch.testing.assert_close(first,second)

    def test_gradient_mask_and_detached_teacher(self):
        student = torch.randn(3,6,8,requires_grad=True)
        teacher = self.x.clone().requires_grad_()
        loss = relational_clip_loss(student,teacher,self.valid);loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(),0)
        self.assertEqual(student.grad[~self.valid].count_nonzero().item(),0)
        self.assertEqual(student.grad[2].count_nonzero().item(),0)
        self.assertIsNone(teacher.grad)

    def test_degenerate_and_empty_are_finite(self):
        x = torch.ones(3,6,8,requires_grad=True)
        loss = relational_clip_loss(x,self.x,self.valid)
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
        loss = relational_clip_loss(x,self.x,torch.zeros_like(self.valid))
        self.assertEqual(loss.item(),0)

    def test_head_and_auxiliary_encoder_gradient(self):
        m = SignRepTransfer('relational')
        x = torch.randn(3,6,1536,requires_grad=True)
        target = torch.randn(3,6,768,requires_grad=True)
        m.supervision(x,target,self.valid).backward()
        self.assertGreater(m.stats['auxiliary_gradient_norm'],0)
        self.assertTrue(m.stats['auxiliary_gradient_finite'])
        self.assertGreater(m.project[1].weight.grad.abs().sum().item(),0)
        self.assertIsNone(target.grad)

    def test_pointwise_recipe_unchanged(self):
        m = SignRepTransfer()
        x = torch.randn(3,6,1536)
        target = torch.randn(3,6,768)
        expected = (1-torch.nn.functional.cosine_similarity(m.project(x)[self.valid],target[self.valid],dim=-1)).mean()
        torch.testing.assert_close(m.supervision(x,target,self.valid),expected,rtol=0,atol=0)

    def test_auxiliary_gradient_scales_with_weight(self):
        m = SignRepTransfer('relational')
        x = torch.randn(3,6,1536,requires_grad=True)
        target = torch.randn(3,6,768)
        (.1*m.supervision(x,target,self.valid)).backward()
        old = x.grad.clone()
        old_norm = m.stats['auxiliary_gradient_norm']
        x.grad = None
        m.zero_grad(set_to_none=True)
        m.supervision(x,target,self.valid).backward()
        torch.testing.assert_close(x.grad,10*old)
        self.assertAlmostEqual(m.stats['auxiliary_gradient_norm']/old_norm,10,places=4)


if __name__=='__main__':unittest.main()
