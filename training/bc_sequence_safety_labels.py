"""Post-sampling label-quality description, never a change to acceptance or training."""
import json
from pathlib import Path

from training.bc_sequence_safety import RUN,ROOT,read_blob,read_record,record,digest


def compute():
    dataset=read_record(RUN/'dataset.json');paths=sorted((RUN/'counterfactuals').glob('*.pkl'))
    if len(paths)!=dataset['accepted']+dataset['rejected']:raise RuntimeError('Incomplete fixed counterfactual set')
    result=dict(source_sha256=digest(Path(__file__)),dataset_sha256=digest(RUN/'dataset.json'),
        all_states=len(paths),bc_deaths=0,student_deaths=0,accepted_student_deaths=0,
        accepted_clear_height_advantage=0,accepted_without_clear_local_advantage=0,
        accepted_higher_bc_common_height=0,accepted=dataset['accepted'],rejected=dataset['rejected'])
    for path in paths:
        r=read_blob(path);a,b=r['branches']['bc'],r['branches']['student'];n=min(len(a['rows']),len(b['rows']))
        result['bc_deaths']+=a['death'];result['student_deaths']+=b['death']
        if r['accepted']:
            result['accepted_student_deaths']+=b['death']
            advantage=not a['death'] and (b['death'] or a['rows'][n-1]['height']<=b['rows'][n-1]['height']-3)
            result['accepted_clear_height_advantage']+=advantage
            result['accepted_without_clear_local_advantage']+=not advantage
            result['accepted_higher_bc_common_height']+=a['rows'][n-1]['height']>b['rows'][n-1]['height']
    record(RUN/'label_quality.json',result);return result


def append():
    result=compute();path=ROOT/'reports/experiments/bc_sequence_safety_results.md';marker='\n## 局部恢复标签的解释边界\n'
    text=path.read_text().split(marker)[0]
    text+=marker+'\n'+f'全部预定反事实：BC真死亡 {result["bc_deaths"]}，学生真死亡 {result["student_deaths"]}。'
    text+=f'接受的 {result["accepted"]} 个标签中，学生局部死亡 {result["accepted_student_deaths"]}；'
    text+=f'BC在共同窗口末高至少低3层或学生已死亡的 {result["accepted_clear_height_advantage"]} 个；'
    text+=f'没有该明确相对优势的 {result["accepted_without_clear_local_advantage"]} 个。\n\n'
    text+='“BC能够恢复”不是“学生在此必然危险”，也不证明BC的首动作是安全最优。以上是事后描述性诊断，不修改原接受规则、样本、训练或开发门槛。'
    text+='出生空间紧张快照为0，不能声称本轮直接覆盖了这类风险。所有被拒绝样本永久保留。\n'
    path.write_text(text)
    path=ROOT/'reports/experiments/bc_sequence_safety_evidence.json';evidence=json.loads(path.read_text());evidence['local_label_quality']=result
    path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    path=ROOT/'reports/experiments/bc_sequence_safety_final.md';text=path.read_text();marker='\n样本覆盖限制：'
    text=text.split(marker)[0]+marker+'预定快照没有出生空间紧张状态；局部恢复成功也不代表相对学生更安全。'
    text+='这些限制不能通过事后补抽或重新解释同一开发组消除。见完整结果中的标签质量诊断。\n';path.write_text(text)


if __name__=='__main__':
    import sys
    if '--append' in sys.argv:append()
    else:print(json.dumps(compute(),ensure_ascii=False))
