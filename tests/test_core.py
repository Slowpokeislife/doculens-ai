import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Chunk, compare_chunks

def test_changes():
    a = [Chunk('a.pdf', 1, 'El plazo de pago establecido será de treinta días naturales.', 0),
         Chunk('a.pdf', 1, 'El servicio incluye asistencia técnica de lunes a viernes.', 1)]
    b = [Chunk('b.pdf', 1, 'El plazo de pago establecido será de sesenta días naturales.', 0),
         Chunk('b.pdf', 1, 'El servicio incluye asistencia técnica de lunes a viernes.', 1),
         Chunk('b.pdf', 2, 'El proveedor añadirá un informe mensual de desempeño.', 2)]
    changes = compare_chunks(a,b)
    assert any(x['type'] == 'Modificado' for x in changes)
    assert any(x['type'] == 'Añadido' for x in changes)
    assert not any(x['type'] == 'Eliminado' for x in changes)


def test_retrieve_comparison_preserves_both_versions(monkeypatch):
    import numpy as np
    import core
    old = [core.Chunk('2025.pdf', 5, 'Disponibilidad mensual 99.5% y primera respuesta crítica 4 horas', 0),
           core.Chunk('2025.pdf', 3, 'La penalización por terminación es 5% del saldo', 1)]
    new = [core.Chunk('2026.pdf', 5, 'Disponibilidad mensual 99.9% y primera respuesta crítica 2 horas', 0),
           core.Chunk('2026.pdf', 3, 'La penalización por terminación es 12% del saldo', 1)]
    changes = core.compare_chunks(old, new)
    monkeypatch.setattr(core, 'embed', lambda texts, model: np.array([[1., 0.]], dtype=np.float32))
    evidence = core.retrieve_comparison('¿Cómo cambiaron los niveles de servicio y las penalizaciones?',
                                        changes, old + new, np.tile([[1.,0.]], (4,1)), 'mock')
    texts = ' '.join(c.text for c, _ in evidence)
    for value in ('99.5%', '99.9%', '5%', '12%'):
        assert value in texts
    assert {c.doc for c, _ in evidence} == {'2025.pdf', '2026.pdf'}


def test_report_markdown_is_rendered():
    import fitz
    import core
    raw = core.make_report('2025.pdf', '2026.pdf', [], '- **Disponibilidad**: mejoró.')
    with fitz.open(stream=raw, filetype='pdf') as pdf:
        text = ''.join(page.get_text() for page in pdf)
    assert '**' not in text
    assert 'Disponibilidad' in text


def test_corporate_pdf_extracts_complete_audit_clause():
    import core
    sample = Path(__file__).resolve().parents[1] / 'demo' / 'Acuerdo_Servicios_Gestionados_2026.pdf'
    chunks, pages = core.read_pdf(sample.read_bytes(), sample.name)
    assert pages == 6
    assert any('auditoría documental' in c.text and 'diez días hábiles' in c.text
               and 'una vez por año' in c.text for c in chunks)
    assert any('PROVEEDOR informará al CLIENTE' in c.text
               and 'quince días naturales' in c.text
               and 'objeciones razonadas' in c.text for c in chunks)


def test_real_demo_comparison_finds_important_changes():
    import core
    root = Path(__file__).resolve().parents[1] / 'demo'
    old, _ = core.read_pdf((root / 'Acuerdo_Servicios_Gestionados_2025.pdf').read_bytes(), '2025.pdf')
    new, _ = core.read_pdf((root / 'Acuerdo_Servicios_Gestionados_2026.pdf').read_bytes(), '2026.pdf')
    changes = core.compare_chunks(old, new)
    paired = [(x['old'].text, x['new'].text) for x in changes if x['old'] and x['new']]
    assert any('99.5%' in a and '99.9%' in b for a, b in paired)
    assert any('4 horas' in a and '2 horas' in b for a, b in paired)
    assert any('185,000' in a and '198,000' in b for a, b in paired)


def test_ollama_invalid_source_citation_warning(monkeypatch):
    import core
    class FakeResponse:
        ok = True
        def json(self):
            return {'message': {'content': 'La cláusula cambió [FUENTE 55].'}}
    monkeypatch.setattr(core.requests, 'post', lambda *args, **kwargs: FakeResponse())
    answer = core.ask_ollama('¿Qué cambió?', [(core.Chunk('test.pdf', 1, 'Cláusula', 0), 1)], 'mock')
    assert 'Advertencia' in answer
