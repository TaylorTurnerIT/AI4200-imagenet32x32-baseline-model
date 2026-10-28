"""Create a self-contained comparison page from completed experiment artifacts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--curves', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    curves = json.loads(args.curves.read_text())
    headline = ('Extra block improves validation accuracy in this trial' if summary['primary_success']
                else 'Extra block did not improve validation accuracy')
    payload = {'summary': summary, 'curves': curves}
    html = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font:14px var(--font-sans,system-ui);color:var(--foreground,#20252b)}p{margin:0 0 10px;line-height:1.45}.muted,footer{color:var(--muted-foreground,#637180);font-size:12px}strong{font-size:16px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:14px 0}.stat{padding:16px;background:var(--secondary,#f3f5f6);border-radius:var(--radius,8px)}.score{display:block;font-size:25px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}.controls{display:flex;gap:14px;align-items:center;margin:14px 0}select{font:inherit;color:inherit;background:var(--secondary,#f3f5f6);border:1px solid var(--border,#ccd2da);border-radius:5px;padding:5px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px}.key{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:5px;background:var(--chart-1,#246baf)}.key.candidate{background:var(--chart-2,#c77c20)}svg{display:block;width:100%;height:260px}footer{line-height:1.5;margin-top:8px}.detail{margin-top:4px;font-size:12px;font-weight:normal;color:var(--muted-foreground,#637180)}
</style></head><body>
<p><strong>__HEADLINE__</strong><br><span class="muted">One fresh candidate attempt · seed 0 · same ImageNet holdout and recipe · no test-set evaluation</span></p>
<div class="stats"><div class="stat">Baseline<span class="score" id="baseline-score"></span><div class="detail" id="baseline-detail"></div></div><div class="stat">Extra block<span class="score" id="candidate-score"></span><div class="detail" id="candidate-detail"></div></div></div>
<p id="difference"></p><div class="controls"><label>Validation <select id="metric"><option value="validation_accuracy">accuracy (%)</option><option value="validation_loss">loss</option></select></label></div>
<div class="legend"><span><i class="key"></i>Baseline</span><span><i class="key candidate"></i>Extra block</span></div><svg role="img" aria-label="Baseline and candidate validation learning curves"></svg>
<footer>Added one 256→512→256 residual block before the final 2×2→1×1 reduction; +263,424 parameters. Existing weights and shuffle RNG stayed seed-matched. Both runs use fresh weights and the same 25-epoch / 60-minute limits. One seed does not establish a general conclusion about depth. Current submission defaults remain unchanged.</footer>
<script>const DATA=__DATA__;const S=DATA.summary;const C=DATA.curves;
document.querySelector('#baseline-score').textContent=(100*S.baseline_best_validation_accuracy).toFixed(4)+'%';document.querySelector('#candidate-score').textContent=(100*S.best_validation_accuracy).toFixed(4)+'%';
document.querySelector('#baseline-detail').textContent='best validation · '+(S.baseline_training_seconds/60).toFixed(2)+' min';document.querySelector('#candidate-detail').textContent='best validation · '+(S.training_seconds/60).toFixed(2)+' min';document.querySelector('#difference').textContent='Candidate change: '+(S.validation_gain_percentage_points>=0?'+':'')+S.validation_gain_percentage_points.toFixed(4)+' percentage points. Selected epoch '+S.selected_epoch+' of '+S.epochs_completed+'.';
const svg=document.querySelector('svg'),ns='http://www.w3.org/2000/svg';function element(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e}
function draw(){svg.replaceChildren();const w=Math.max(300,svg.clientWidth),h=260,L=44,R=12,T=18,B=32,metric=document.querySelector('#metric').value,factor=metric==='validation_accuracy'?100:1,all=[...C.baseline,...C.candidate].map(r=>factor*r[metric]),lo=Math.min(...all),hi=Math.max(...all),pad=Math.max((hi-lo)*.1,.01),low=lo-pad,high=hi+pad;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);const x=e=>L+(e-1)/24*(w-L-R),y=v=>h-B-(v-low)/(high-low)*(h-T-B);
for(let i=0;i<=4;i++){const value=low+(high-low)*i/4,py=y(value);element('line',{x1:L,y1:py,x2:w-R,y2:py,stroke:'var(--border,#dfe3e7)'});element('text',{x:L-7,y:py+4,'text-anchor':'end',fill:'var(--muted-foreground,#637180)','font-size':11},value.toFixed(metric==='validation_accuracy'?1:2))}for(const epoch of [1,5,10,15,20,25])element('text',{x:x(epoch),y:h-13,'text-anchor':'middle',fill:'var(--muted-foreground,#637180)','font-size':11},epoch);element('text',{x:w-R,y:h-1,'text-anchor':'end',fill:'var(--muted-foreground,#637180)','font-size':10},'Epoch');
for(const[rows,color,label]of[[C.baseline,'var(--chart-1,#246baf)','Baseline'],[C.candidate,'var(--chart-2,#c77c20)','Extra block']]){element('path',{d:rows.map((r,i)=>(i?'L':'M')+x(r.epoch)+','+y(factor*r[metric])).join(' '),fill:'none',stroke:color,'stroke-width':2});for(const row of rows){const dot=element('circle',{cx:x(row.epoch),cy:y(factor*row[metric]),r:3,fill:color});const title=document.createElementNS(ns,'title');title.textContent=label+' epoch '+row.epoch+': '+(factor*row[metric]).toFixed(4);dot.append(title)}}}
document.querySelector('#metric').addEventListener('change',draw);new ResizeObserver(draw).observe(svg);draw();console.assert(C.candidate.length===S.epochs_completed,'Candidate epoch count mismatch');
</script></body></html>'''
    args.output.write_text(html.replace('__HEADLINE__',headline).replace('__DATA__',json.dumps(payload)))


if __name__ == '__main__':
    main()
