"""Same fixed UPRet bank: source max reduction versus transport-weighted sum."""
import ast
import hashlib
import json
import subprocess
from types import MethodType, SimpleNamespace

import torch

from .common import ROOT, dump, sha
from .upret_gradient_audit import COMMIT, OUT, reduction, analytic_plan


def main():
    output = OUT/'UPRET-REDUCTION_run.json'
    if output.exists():
        raise FileExistsError(output)
    original_path = OUT/'UPRET-GRADIENT_run.json'
    original = json.loads(original_path.read_text())
    source = subprocess.check_output(['git','-C',str(ROOT/'third_party/UPRet'),
                                      'show',COMMIT+':modules/modeling.py']).decode()
    assert hashlib.sha256(source.encode()).hexdigest() == original['source_sha256']
    sink = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='Sinkhorn')
    scope = {'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[sink],type_ignores=[])),'<upret-Sinkhorn>','exec'),scope)
    model = SimpleNamespace(max_iter=100)
    model.Sinkhorn = MethodType(scope['Sinkhorn'],model)
    scores = []
    for item in original['entries']:
        sim = torch.tensor([item['sim']],dtype=torch.float64)
        mass = torch.full((1,2),.5,dtype=sim.dtype)
        plan = model.Sinkhorn(torch.exp(-(1-sim)/.1),mass,mass)
        source_max = float(reduction(sim,plan).item())
        assert abs(source_max-item['source_score']) < 1e-14
        exact = analytic_plan(sim)
        scores.append({'source_max':source_max,'source_plan_sum':float((sim*plan).sum()),
                       'exact_max':float(reduction(sim,exact).item()),
                       'exact_plan_sum':float((sim*exact).sum())})
    reversals = {}
    for prefix in ('source','exact'):
        maximum_key = prefix+'_max'
        sum_key = prefix+'_plan_sum'
        reversals[prefix] = [[i,j] for i in range(len(scores)) for j in range(i+1,len(scores))
                             if (scores[i][maximum_key]-scores[j][maximum_key])*
                                (scores[i][sum_key]-scores[j][sum_key]) < 0]
    result = {'status':'completed','code_sha256':sha(__file__),'source_run_sha256':sha(original_path),
              'source_reduction_reconstructed_tolerance':1e-14,'score_count':len(scores),
              'pair_comparisons':len(scores)*(len(scores)-1)//2,
              'strict_reversal_counts':{k:len(v) for k,v in reversals.items()},
              'strict_reversal_indexes':reversals,'scores':scores,
              'scope':'posthoc same synthetic bank; not retrieval ranking, no new cases selected',
              'test_loaded':False,'assets_loaded':False,'method_go':False}
    dump(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('scores','strict_reversal_indexes')},indent=2))


if __name__=='__main__':
    main()
