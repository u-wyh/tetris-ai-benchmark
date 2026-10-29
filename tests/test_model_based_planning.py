from training.model_based_planning import gate

def result():
    a=dict(games=16,mean_pieces=20000,p10=20000,early=0,capped=16,mean_score=100,high_fraction=0,new_holes_per_100=2,mean_max_high_streak=0,p95_ms=49)
    return dict(summary=dict(bc=a,student=dict(a),planner=dict(a)),paired=dict(planner=dict(common_score_ratio=1.10,positive_seeds=13,ratio_ci95=[1.08,1.12])))

def test_short_gate_score_student_survival_and_cpu_are_required():
    r=result();assert gate(r)['passed']
    r['paired']['planner']['common_score_ratio']=1.099;assert not gate(r)['passed']
    r=result();r['summary']['planner']['p95_ms']=50.01;assert not gate(r)['passed']
    r=result();r['summary']['planner']['early']=1;assert not gate(r)['passed']
    r=result();r['summary']['planner']['mean_pieces']=19000;assert not gate(r)['passed']

def test_long_gate_total_score_and_confirmation_ci():
    r=result();assert gate(r,long=True,confirm=True)['passed']
    r['summary']['planner']['mean_score']=99;assert not gate(r,long=True)['passed']
    r=result();r['paired']['planner']['ratio_ci95']=[.99,1.2];assert not gate(r,long=True,confirm=True)['passed']
