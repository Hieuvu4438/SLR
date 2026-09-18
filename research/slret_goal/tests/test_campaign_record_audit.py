from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from campaign_record_audit import compare_record


def test_status_mismatch_not_hidden_by_summary_schema():
    report=dict(status='completed',value=1)
    assert compare_record(report,None)['ledger_missing']
    assert not compare_record(report,dict(status='running',value=1))['status_match']
    compared=compare_record(report,dict(status='completed',artifact='path',value=2))
    assert compared['status_match']
    assert compared['extra_fields']==['artifact']
    assert compared['field_differences']==['value']
