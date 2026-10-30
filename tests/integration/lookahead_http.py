"""Bounded live HTTP boundary/concurrency probes; not a gameplay benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
from training.tetris_core.core import TetrisCore

p=argparse.ArgumentParser();p.add_argument('--host',default='100.91.227.47');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
url=f'http://{args.host}:8081';origin=f'http://{args.host}:8080'
core=TetrisCore(3102841030);public=core.get_public_observation();legal=[x['actionId'] for x in core.get_legal_placements()]
body=dict(mode='lookahead',observation=public,legalActionIds=legal)
def request(data=None,path='/api/decide',method=None,origin_value=origin):
    r=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),method=method,headers={'Origin':origin_value,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=10) as response:return response.status,response.headers.get('Access-Control-Allow-Origin'),response.read()
    except urllib.error.HTTPError as e:return e.code,e.headers.get('Access-Control-Allow-Origin'),e.read()
checks={}
checks['preflight']=request(method='OPTIONS')[0];assert checks['preflight']==204
checks['bad_origin']=request(path='/api/health',method='GET',origin_value='http://invalid.example')[0];assert checks['bad_origin']==403
for name,data in [('hidden_bag',{**body,'observation':{**public,'bag':['I']}}),('next4',{**body,'observation':{**public,'next':public['next']+['I']}}),('wrong_mask',{**body,'legalActionIds':[1,2]}),('missing_mask',dict(mode='lookahead',observation=public))]:
    checks[name]=request(data)[0];assert checks[name]==400,name
barrier=threading.Barrier(8)
def parallel(_):barrier.wait();return request(body)
with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(parallel,range(8)))
statuses=[r[0] for r in rows];assert all(s in (200,503) for s in statuses)
assert all(r[1]==origin for r in rows)
accepted=[json.loads(r[2]) for r in rows if r[0]==200]
assert accepted and len({r['actionId'] for r in accepted})==1
result=dict(checks=checks,concurrent_requests=8,statuses=statuses,accepted_same_action=True,health=request(path='/api/health',method='GET')[0])
assert result['health']==200
args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(args.output);print(json.dumps(result))
