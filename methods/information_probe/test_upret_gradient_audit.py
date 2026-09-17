import torch
from .upret_gradient_audit import analytic_plan, central, reduction


def test_exact_plan_has_uniform_marginals():
    sim = torch.tensor([[[.1,.3],[-.2,.4]]],dtype=torch.float64)
    plan = analytic_plan(sim)
    assert torch.allclose(plan.sum(1),torch.full((1,2),.5,dtype=torch.float64))
    assert torch.allclose(plan.sum(2),torch.full((1,2),.5,dtype=torch.float64))
    ratio = plan[0,0,0]*plan[0,1,1]/(plan[0,0,1]*plan[0,1,0])
    assert torch.allclose(ratio,torch.exp((sim[0,0,0]+sim[0,1,1]-sim[0,0,1]-sim[0,1,0])/.1))


def test_exact_full_gradient_matches_finite_differences():
    sim = torch.tensor([[[.1,.3],[-.2,.4]]],dtype=torch.float64,requires_grad=True)
    fn = lambda z: reduction(z,analytic_plan(z))
    grad, = torch.autograd.grad(fn(sim).sum(),sim)
    assert torch.allclose(grad,central(fn,sim,1e-6),atol=1e-9,rtol=1e-9)
