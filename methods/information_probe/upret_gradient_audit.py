"""Execute committed UPRet transport block on realizable synthetic similarities."""
import ast
import copy
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace, MethodType

import torch

from .common import ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
COMMIT = '046366227417e1d8ec14145965403462df345984'


def reduction(sim, plan):
    weighted = sim*plan
    return (weighted.max(-2).values.mean(-1) + weighted.max(-1).values.mean(-1))/2


def central(fn, x, step):
    values = []
    for index in range(x.numel()):
        plus, minus = x.detach().clone(), x.detach().clone()
        plus.flatten()[index] += step
        minus.flatten()[index] -= step
        values.append((fn(plus).sum()-fn(minus).sum())/(2*step))
    return torch.stack(values).reshape_as(x)


def analytic_plan(sim):
    contrast = sim[:,0,0]+sim[:,1,1]-sim[:,0,1]-sim[:,1,0]
    diagonal = .5*torch.sigmoid(contrast/.2)
    return torch.stack([diagonal,.5-diagonal,.5-diagonal,diagonal], -1).reshape(-1,2,2)


def main():
    output = OUT/'UPRET-GRADIENT_run.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    source = subprocess.check_output(['git','-C',str(ROOT/'third_party/UPRet'),
                                      'show',COMMIT+':modules/modeling.py']).decode()
    tree = ast.parse(source)
    sink = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='Sinkhorn')
    scorer = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='flip_similarity_softmax')
    branch = next(n for n in scorer.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self.training')
    begin = next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='wdist')
    wrapper = ast.parse('def source_score(self, sim):\n    B_v, B_t = 1, sim.shape[0]\n').body[0]
    wrapper.body += copy.deepcopy(branch.body[begin:])
    wrapper.body += ast.parse('return sim_ot').body
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(sink),wrapper],type_ignores=[]))
    scope = {'torch':torch}
    exec(compile(module,'<committed-upret-transport>','exec'), scope)
    model = SimpleNamespace(sample_num=2,eps=.1,max_iter=100)
    model.Sinkhorn = MethodType(scope['Sinkhorn'],model)
    score = MethodType(scope['source_score'],model)
    torch.set_num_threads(2)
    torch.manual_seed(20260915)
    bank = torch.rand(32,2,2,dtype=torch.float64)-.5
    # Construct explicit unit embeddings proving all tested matrices realizable.
    v = torch.eye(4,dtype=torch.float64)[:2]
    t = torch.zeros(32,2,4,dtype=torch.float64)
    t[:,:,:2] = bank.transpose(-1,-2)
    t[:,:,2] = (1-t[:,:,:2].square().sum(-1)).sqrt()
    assert torch.allclose(t.norm(dim=-1),torch.ones(32,2,dtype=torch.float64),atol=1e-14)
    assert torch.equal(torch.einsum('id,bjd->bij',v,t),bank)
    entries = []
    for item in bank:
        x = item[None].clone().requires_grad_()
        y = score(x)
        grad, = torch.autograd.grad(y.sum(),x)
        with torch.no_grad():
            masses = torch.full((1,2),.5,dtype=x.dtype)
            plan = model.Sinkhorn(torch.exp(-(1-x)/.1),masses,masses)
        assert torch.equal(y.flatten(),reduction(x,plan))
        fixed_fd = central(lambda z: reduction(z,plan),x,1e-6)
        fd1,fd2 = central(score,x,1e-5),central(score,x,1e-6)
        analytic_x = x.detach().clone().requires_grad_()
        analytic_y = reduction(analytic_x,analytic_plan(analytic_x))
        analytic_grad, = torch.autograd.grad(analytic_y.sum(),analytic_x)
        analytic_fixed = x.detach().clone().requires_grad_()
        frozen_y = reduction(analytic_fixed,analytic_plan(analytic_fixed).detach())
        frozen_grad, = torch.autograd.grad(frozen_y.sum(),analytic_fixed)
        assert float((grad-fixed_fd).abs().max()) < 1e-8
        entries.append({'sim':item.tolist(),'source_score':float(y.detach().sum()),
                        'autograd':grad.detach().tolist(),'finite_difference':fd2.detach().tolist(),
                        'fixed_plan_fd_max_error':float((grad-fixed_fd).abs().max()),
                        'fd_step_stability_max_error':float((fd1-fd2).abs().max()),
                        'source_gradient_discrepancy_max':float((grad-fd2).abs().max()),
                        'source_plan_vs_exact_max_error':float((plan-analytic_plan(x.detach())).abs().max()),
                        'exact_plan_full_vs_frozen_gradient_discrepancy_max':float((analytic_grad-frozen_grad).abs().max())})
    result = {'status':'completed','commit':COMMIT,'code_sha256':sha(__file__),
              'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'extracted_ast_sha256':hashlib.sha256(ast.dump(module).encode()).hexdigest(),
              'protocol_sha256':sha(OUT/'UPRET_gradient_protocol.md'),
              'synthetic_count':len(entries),'realizable_unit_embeddings':True,
              'cases_source_gradient_difference_gt_1e_5':sum(e['source_gradient_discrepancy_max']>1e-5 for e in entries),
              'max_source_gradient_discrepancy':max(e['source_gradient_discrepancy_max'] for e in entries),
              'max_fd_step_instability':max(e['fd_step_stability_max_error'] for e in entries),
              'max_fixed_plan_control_error':max(e['fixed_plan_fd_max_error'] for e in entries),
              'max_exact_plan_gradient_discrepancy':max(e['exact_plan_full_vs_frozen_gradient_discrepancy_max'] for e in entries),
              'cases_exact_plan_gradient_difference_gt_1e_5':sum(e['exact_plan_full_vs_frozen_gradient_discrepancy_max']>1e-5 for e in entries),
              'entries':entries,'test_loaded':False,'assets_loaded':False,'method_go':False,
              'wall_seconds':time.monotonic()-started}
    dump(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='entries'},indent=2))


if __name__=='__main__':
    main()
