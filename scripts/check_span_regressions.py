"""Check predeclared development criteria after frozen checkpoint comparison."""
import argparse
import json
from pathlib import Path
from privai_model.ml_span.regression import check_regressions
from privai_model.io import write_json

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--report',required=True);p.add_argument('--gates',required=True)
 p.add_argument('--output',required=True);p.add_argument('--model',default='candidate')
 a=p.parse_args()
 result=check_regressions(json.loads(Path(a.report).read_text()),json.loads(Path(a.gates).read_text())['gates'],a.model)
 write_json(a.output,result)
 print(json.dumps(result))
