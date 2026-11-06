"""Build the local website's result tables and example metadata."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    tables=json.loads((ROOT/'results/reproduced/tables.json').read_text(encoding='utf-8'))
    published=json.loads((ROOT/'configs/paper_tables.json').read_text(encoding='utf-8'))
    benchmarks={}
    for key,table,title,unit,note in [
        ('abstract','abstract_stage3','Abstract 2D · False belief','Accuracy (%)','500 probes per model · Table 3'),
        ('muma','muma','MuMA-ToM','Accuracy (%)','900 questions per model · Table 4'),
        ('threedworld','mindpower_threedworld','MindPower · ThreeDWorld','Layer-averaged score','36 scenarios per model · Table 5a'),
        ('virtualhome','mindpower_virtualhome','MindPower · VirtualHome','Layer-averaged score','456 scenarios per model · Table 5b')]:
        rows=[dict(row,model=tables[table][index]['model']) for index,row in enumerate(published[table])]
        benchmarks[key]={'title':title,'unit':unit,'note':note,'conditions':[c for c in rows[0] if c!='model'],'rows':rows}
    probes={p['id']:p for p in json.loads((ROOT/'data/abstract2d/probes.json').read_text(encoding='utf-8'))}
    examples=[]
    for pid,label in [('s3_two_rooms_000','False belief · Two rooms'),('s3_sally_anne_reality_000','False belief · Sally–Anne'),('s1_occlusion_consistent_000','Object tracking · Occlusion'),('s2_wall_removed_000','Goal attribution · Wall removed')]:
        p=probes[pid]
        examples.append({'id':pid,'label':label,'question':p['question_A'],'answer':p['expected'],'description':p['text_narration'],'panel_count':p['panel_count']})
        shutil.copy2(ROOT/p['image_path'],ROOT/'website/assets'/f'{pid}.png')
    (ROOT/'website/data.js').write_text('window.PAPER_DATA = '+json.dumps({'benchmarks':benchmarks,'examples':examples},ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    print('Website tables use the printed paper values. Recomputed values remain in results/reproduced/.')


if __name__=='__main__':main()
