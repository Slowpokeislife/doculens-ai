from pathlib import Path
import sys
import fitz
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core

ROOT=Path(__file__).resolve().parents[1]/'demo'

def load_pair(prefix):
    old=ROOT/f'{prefix}_2025.pdf'
    new=ROOT/f'{prefix}_2026.pdf'
    return core.read_pdf(old.read_bytes(),old.name)[0], core.read_pdf(new.read_bytes(),new.name)[0]


def test_exactly_two_demo_pairs():
    assert sorted(p.name for p in ROOT.glob('*.pdf'))==sorted([
        'Acuerdo_Servicios_Gestionados_2025.pdf','Acuerdo_Servicios_Gestionados_2026.pdf',
        'Managed_Services_Agreement_2025.pdf','Managed_Services_Agreement_2026.pdf'
    ])


def test_auto_detects_both_languages():
    for prefix,expected in [('Acuerdo_Servicios_Gestionados','es'),('Managed_Services_Agreement','en')]:
        a,b=load_pair(prefix)
        assert core.detect_language(' '.join(x.text for x in a+b)) == expected


def test_english_pdf_and_important_changes():
    a,b=load_pair('Managed_Services_Agreement')
    changes=core.compare_chunks(a,b)
    paired=[(c['old'].text,c['new'].text) for c in changes if c['old'] and c['new']]
    assert any('99.5%' in x and '99.9%' in y for x,y in paired)
    assert any('4 hours' in x and '2 hours' in y for x,y in paired)
    assert any('$185,000' in x and '$198,000' in y for x,y in paired)
    report=core.make_report('before.pdf','after.pdf',changes,'- **Availability** improved.',language='en')
    with fitz.open(stream=report,filetype='pdf') as doc:
        text=' '.join(p.get_text() for p in doc)
    assert 'Executive Change Report' in text
    assert 'Total changes' in text
    assert 'AI-generated summary' in text
    assert 'Detailed change appendix' in text
    assert 'Se modificó' not in text


def test_spanish_report_headings():
    a,b=load_pair('Acuerdo_Servicios_Gestionados')
    report=core.make_report('antes.pdf','despues.pdf',core.compare_chunks(a,b),language='es')
    with fitz.open(stream=report,filetype='pdf') as doc:
        text=' '.join(p.get_text() for p in doc)
    assert 'Informe ejecutivo de cambios' in text
    assert 'Cambios totales' in text


def test_prompts_enforce_requested_language(monkeypatch):
    captures=[]
    class Response:
        ok=True
        def json(self):return {'message':{'content':'Answer [SOURCE 1].'}}
    def fake_post(*args,**kwargs):
        captures.append(kwargs['json'])
        return Response()
    monkeypatch.setattr(core.requests,'post',fake_post)
    chunks=[(core.Chunk('doc.pdf',1,'Availability 99.9%',0),1.)]
    assert 'Answer' in core.ask_ollama('What changed?',chunks,'model',language='en')
    assert 'ENTIRELY in English' in captures[-1]['messages'][0]['content']
    core.ask_ollama('¿Qué cambió?',chunks,'model',language='es')
    assert 'COMPLETAMENTE en español' in captures[-1]['messages'][0]['content']
