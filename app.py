"""DocuLens AI v0.5 — Dimension Edition, bilingual local document intelligence."""
from __future__ import annotations
from html import escape
from pathlib import Path
import hashlib
import streamlit as st
from core import (read_pdf, compare_chunks, ollama_models, embed, retrieve_comparison,
                  ask_ollama, make_report, diff_tokens_html, top_findings, change_severity,
                  detect_language)
from i18n import TRANSLATIONS, localize_type, localize_severity

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='DocuLens — Document Intelligence', page_icon=str(ROOT/'assets/logo.svg'),
                   layout='wide', initial_sidebar_state='collapsed')
st.markdown(f"<style>{(ROOT/'assets/style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def safe(value):
    return escape(str(value), quote=True)


def go(screen):
    st.session_state.screen = screen
    st.session_state.navigation = screen


def navigation_changed():
    st.session_state.screen = st.session_state.navigation


def reset_answers():
    st.session_state.analysis = ''
    st.session_state.qa = None


def use_demo(lang):
    prefix = ('Acuerdo_Servicios_Gestionados' if lang=='es' else 'Managed_Services_Agreement')
    folder = ROOT/'demo'
    a = folder/f'{prefix}_2025.pdf'
    b = folder/f'{prefix}_2026.pdf'
    st.session_state.document_pair=(a.read_bytes(),a.name,b.read_bytes(),b.name)
    st.session_state.document_source='demo'
    if st.session_state.language_choice == 'auto':
        st.session_state.detected_lang = lang
    reset_answers()
    go('Workspace')


def section(index, title, description='',tight=False):
    st.markdown(f'<section class="dl-section {"tight" if tight else ""}"><div class="index">{safe(index)}</div>'
                f'<h2>{safe(title)}</h2><p>{safe(description)}</p></section>',unsafe_allow_html=True)


def tag(value):
    return f'<span class="dl-tag">{safe(value)}</span>'


def metric(label, value, subtitle):
    st.markdown(f'<div class="dl-card"><div class="dl-card-label">{safe(label)}</div>'
                f'<div class="dl-number">{safe(value)}</div><div class="dl-number-sub">{safe(subtitle)}</div></div>',
                unsafe_allow_html=True)


def refresh_comparison(t):
    pair=st.session_state.get('document_pair')
    if not pair:return False
    a,a_name,b,b_name=pair
    identity=(hashlib.sha256(a).hexdigest(),a_name,hashlib.sha256(b).hexdigest(),b_name)
    if st.session_state.get('identity')==identity:return True
    try:
        with st.spinner(t['reading']):
            old,p_old=read_pdf(a,a_name);new,p_new=read_pdf(b,b_name)
            if not old or not new:
                st.error(t['extraction_error']);return False
            changes=compare_chunks(old,new)
            st.session_state.update(identity=identity,old_chunks=old,new_chunks=new,
                                    pages=(p_old,p_new),changes=changes,
                                    findings=top_findings(changes,limit=6,language=lang),analysis='',qa=None,
                                    matrix=None,matrix_model=None)
        return True
    except Exception as exc:
        st.error(f"{t['compare_error']} {exc}");return False


def file_upload(t):
    st.markdown(f'<div class="dl-mini-title">{safe(t["upload_label"])}</div>',unsafe_allow_html=True)
    x,y=st.columns(2,gap='large')
    with x:old=st.file_uploader(t['old_upload'],type=['pdf'],key='upload_old')
    with y:new=st.file_uploader(t['new_upload'],type=['pdf'],key='upload_new')
    if old and new:
        pair=(old.getvalue(),old.name,new.getvalue(),new.name)
        if st.session_state.get('document_pair')!=pair:
            st.session_state.document_pair=pair
            reset_answers()
    elif old or new:st.info(t['upload_missing'])


def document_strip(t):
    pair=st.session_state.document_pair
    x,y=st.columns(2,gap='medium')
    for col,blob,name,label in ((x,pair[0],pair[1],t['before']),
                                (y,pair[2],pair[3],t['after'])):
        with col:
            st.markdown(f'<div class="dl-file"><div class="file-icon">▤</div><div><b>{safe(name)}</b>'
                        f'<small>{safe(label)} · {len(blob)/1024:.0f} KB</small></div></div>',unsafe_allow_html=True)


def stat_row(t):
    changes=st.session_state.changes
    counts={k:sum(c['type']==k for c in changes) for k in ('Modificado','Añadido','Eliminado')}
    values=[len(changes),counts['Modificado'],counts['Añadido'],counts['Eliminado']]
    for col,label,val,hint in zip(st.columns(4,gap='medium'),t['metric_labels'],values,t['metric_hints']):
        with col:metric(label,val,hint)


def findings_list(t, lang, limit=6):
    items=top_findings(st.session_state.changes,limit=limit,language=lang)
    if not items:st.info(t['no_changes']);return
    for i,item in enumerate(items,1):
        # Heuristic summaries derived from source text may legitimately be in the document language.
        st.markdown('<div class="dl-find">'+f'<div class="dl-find-index">{i:02d}</div>'
                    +f'<div class="dl-find-body"><div class="dl-find-meta">{safe(localize_type(item["type"],lang))} · '
                    +f'{safe(t["priority"])} {safe(localize_severity(item["severity"],lang))} · {safe(t["page"])} {safe(item["page"])}</div>'
                    +f'{safe(item["summary"])}</div></div>',unsafe_allow_html=True)

for key,default in [('screen','Overview'),('language_choice','auto')]:
    if key not in st.session_state:st.session_state[key]=default
if 'navigation' not in st.session_state:st.session_state.navigation=st.session_state.screen

# Automatic mode infers language from the actual PDF text (falls back to Spanish without docs).
if st.session_state.get('document_pair'):
    pair=st.session_state.document_pair
    document_id=hashlib.sha256(pair[0]+pair[2]).hexdigest()
    if st.session_state.get('detected_from')!=document_id:
        try:
            # Sample all pages to avoid a cover page dominating language detection.
            import fitz
            samples=[]
            for blob in (pair[0],pair[2]):
                with fitz.open(stream=blob,filetype='pdf') as pdf:
                    samples.extend(page.get_text()[:2500] for page in pdf)
            st.session_state.detected_lang=detect_language(' '.join(samples))
        except Exception:
            st.session_state.detected_lang='es'
        st.session_state.detected_from=document_id
else:
    st.session_state.detected_lang='es'

lang=st.session_state.detected_lang if st.session_state.language_choice=='auto' else st.session_state.language_choice
t=TRANSLATIONS[lang]

st.markdown('<nav class="dl-navbar"><div class="dl-brand"><span class="dl-mark">⌕</span>DocuLens '
            '<span style="color:#a6a6a6;font-weight:400">AI</span></div>'
            '<div class="dl-navbar-right"><span class="dl-chip">Local-first</span><span class="dl-chip">v0.5 · Bilingual Edition</span></div></nav>',
            unsafe_allow_html=True)

navcol,langcol=st.columns([1.7,1],gap='large',vertical_alignment='center')
with navcol:
    st.radio('Navigation',['Overview','Workspace','Reports'],horizontal=True,
             format_func=lambda x:{'Overview':t['overview'],'Workspace':t['workspace'],'Reports':t['reports']}[x],
             label_visibility='collapsed',key='navigation',on_change=navigation_changed)
with langcol:
    # Use an untranslatable stable value for state, but translate visible labels.
    st.selectbox(t['language'],['auto','es','en'],key='language_choice',
                 format_func=lambda x:{'auto':t['auto'],'es':'Español','en':'English'}[x],on_change=reset_answers)
    if st.session_state.language_choice=='auto':
        st.caption(f"{t['language_auto']}: {'Español' if lang=='es' else 'English'}")

btn1,btn2,btn3=st.columns([1,1,1.15],gap='small')
with btn1:st.button(t['demo_es'],on_click=use_demo,args=('es',),use_container_width=True)
with btn2:st.button(t['demo_en'],on_click=use_demo,args=('en',),use_container_width=True)
with btn3:st.button(t['start'],type='primary',on_click=go,args=('Workspace',),use_container_width=True)

with st.sidebar:
    st.markdown('### '+t['settings'])
    st.caption(t['ollama_help'])
    try:
        models=ollama_models()
        emb_prefix=('nomic-embed','mxbai-embed','all-minilm','snowflake-arctic-embed',
                    'bge-','embeddinggemma','granite-embedding','qwen3-embedding','paraphrase-multilingual')
        emb=[x for x in models if x.lower().startswith(emb_prefix)]
        chats=[x for x in models if x not in emb]
        llm=st.selectbox(t['chat_model'],chats,
                         index=next((i for i,m in enumerate(chats) if m.startswith('llama3.2:3b')),0) if chats else None)
        embedding=st.selectbox(t['embed_model'],emb,
                         index=next((i for i,m in enumerate(emb) if m.startswith('nomic-embed-text')),0) if emb else None)
        if llm and embedding:st.success(t['ready'])
        else:st.info(t['offline'])
    except Exception:
        llm=embedding=None;st.info(t['offline'])
    st.divider();st.markdown('**'+t['limits']+'**');st.caption(t['scope'])

if st.session_state.screen=='Overview':
    st.markdown(f'<section class="dl-hero"><div class="dl-kicker">{safe(t["hero_kicker"])}</div>'
                f'<h1>{safe(t["hero_title"]).replace(chr(10),"<br/>")}</h1><p>{safe(t["hero_text"])}</p>'
                '<div class="dl-hero-points">'+''.join('<span>✓ '+safe(x)+'</span>' for x in t['hero_features'])
                +'</div></section>',unsafe_allow_html=True)
    section('01 / WORKFLOW',t['workflow'],t['workflow_desc'])
    for col,card in zip(st.columns(3,gap='medium'),t['workflow_cards']):
        with col:
            st.markdown(f'<div class="dl-card"><div class="dl-card-label">{safe(card[0])}</div>'
                        f'<h3>{safe(card[1])}</h3><p>{safe(card[2])}</p></div>',unsafe_allow_html=True)
    section('02 / PRODUCT DEMO',t['preview'],t['preview_note'])
    first,second=st.columns([1.5,1],gap='large')
    if lang=='en':
        pair_preview=('Monthly availability','Critical first response','4 hours','2 hours')
    else:pair_preview=('Disponibilidad mensual','Primera respuesta crítica','4 horas','2 horas')
    with first:
        st.markdown(f'<div class="dl-card"><div class="dl-mini-title">{safe(t["preview"])}</div><div class="dl-diff-grid">'
                    f'<div class="dl-diff"><h4>{safe(t["before"])} · 2025</h4><div class="body">{safe(pair_preview[0])} '
                    f'<span class="diff-del">99.5%</span><br/>{safe(pair_preview[1])} <span class="diff-del">{pair_preview[2]}</span></div></div>'
                    f'<div class="dl-diff"><h4>{safe(t["after"])} · 2026</h4><div class="body">{safe(pair_preview[0])} '
                    f'<span class="diff-add">99.9%</span><br/>{safe(pair_preview[1])} <span class="diff-add">{pair_preview[3]}</span></div></div>'
                    f'</div><p style="margin-top:18px;font-size:12px">{safe(t["preview_note"])}</p></div>',unsafe_allow_html=True)
    with second:
        st.markdown('<div class="dl-card"><div class="dl-card-label">LOCAL PROTOTYPE</div>'
                    '<div class="dl-editorial-row"><span>LLM</span><b>Ollama</b></div>'
                    '<div class="dl-editorial-row"><span>Backend</span><b>Python</b></div>'
                    '<div class="dl-editorial-row"><span>UI</span><b>Streamlit</b></div></div>',unsafe_allow_html=True)

elif st.session_state.screen=='Workspace':
    section('01 / WORKSPACE',t['work_title'],t['work_desc'],tight=True)
    with st.expander(t['upload_label'],expanded=not bool(st.session_state.get('document_pair'))):file_upload(t)
    if not st.session_state.get('document_pair'):
        st.info(t['no_docs']);st.stop()
    document_strip(t)
    if not refresh_comparison(t):st.stop()
    section('02 / METRICS',t['metrics_title'],t['metrics_desc'],tight=True)
    stat_row(t)
    section('03 / COMPARISON',t['diff_title'],t['diff_desc'],tight=True)
    selected=st.multiselect(t['filter'],['Modificado','Añadido','Eliminado'],
                            default=['Modificado','Añadido','Eliminado'],
                            format_func=lambda x:localize_type(x,lang))
    changes=[c for c in st.session_state.changes if c['type'] in selected]
    if not changes:st.info(t['none_filter'])
    for i,change in enumerate(changes[:45],1):
        a,b=change['old'],change['new']
        before,after=diff_tokens_html(a.text,b.text) if a and b else (safe(a.text) if a else '—',safe(b.text) if b else '—')
        page_label=f"{t['page']} {a.page} → {b.page}" if a and b else f"{t['page']} {(a or b).page}"
        with st.expander(f'{i:02d} / {localize_type(change["type"],lang)} · {page_label}',expanded=i==1):
            st.markdown(tag(localize_type(change['type'],lang))+tag(page_label)+tag(t['priority']+' '+localize_severity(change_severity(change),lang)),unsafe_allow_html=True)
            st.markdown('<div class="dl-diff-grid">'
                        +f'<div class="dl-diff"><h4>{safe(t["prev"])} · {safe(st.session_state.document_pair[1])}</h4><div class="body">{before}</div></div>'
                        +f'<div class="dl-diff"><h4>{safe(t["next"])} · {safe(st.session_state.document_pair[3])}</h4><div class="body">{after}</div></div>'
                        +'</div>',unsafe_allow_html=True)
    if len(changes)>45:st.caption(t['show_limit'])
    st.markdown('<div class="dl-wash"></div>',unsafe_allow_html=True)
    section('04 / RAG',t['questions_title'],t['questions_desc'],tight=True)
    question=st.text_input(t['question_label'],placeholder=t['question_placeholder'],key='question')
    if st.button(t['ask'],type='primary',disabled=not bool(llm and embedding and question.strip())):
        try:
            chunks=st.session_state.old_chunks+st.session_state.new_chunks
            if st.session_state.matrix is None or st.session_state.matrix_model!=embedding:
                with st.spinner(t['embedding_progress']):
                    st.session_state.matrix=embed([c.text for c in chunks],embedding)
                    st.session_state.matrix_model=embedding
            with st.spinner(t['answer_progress']):
                sources=retrieve_comparison(question,st.session_state.changes,chunks,st.session_state.matrix,embedding)
                answer=ask_ollama(question,sources,llm,language=lang)
                st.session_state.qa=(answer,sources)
        except Exception as exc:st.error(f"{t['error_ai']} {exc}")
    if not llm or not embedding:st.info(t['no_ollama'])
    if st.session_state.get('qa'):
        answer,sources=st.session_state.qa
        st.markdown('### '+t['answer']);st.markdown(answer)
        st.markdown('### '+t['sources'])
        for i,(chunk,score) in enumerate(sources,1):
            with st.expander(f"{t['source']} {i} · {chunk.doc} · {t['page']} {chunk.page} · {score:.2f}"):
                st.write(chunk.text)

elif st.session_state.screen=='Reports':
    section('01 / REPORTS',t['report_title'],t['report_desc'],tight=True)
    if not st.session_state.get('document_pair'):
        st.info(t['empty_report']);st.stop()
    if not refresh_comparison(t):st.stop()
    pair=st.session_state.document_pair
    st.markdown(f'<div class="dl-report-banner"><div class="dl-kicker">DOCUMENT REVIEW / EXECUTIVE BRIEF</div>'
                f'<h2>{safe(t["report_banner_title"])}</h2><p style="color:#aaa">{safe(t["report_banner_text"])}</p></div>',
                unsafe_allow_html=True)
    document_strip(t);stat_row(t)
    left,right=st.columns([1.4,.85],gap='large')
    with left:
        section('02 / FINDINGS',t['key_title'],t['key_desc'],tight=True)
        findings_list(t,lang)
    with right:
        section('03 / DETAILS',t['details_title'],tight=True)
        detail_rows=[(t['details_file'],'2 PDF'),(t['details_pages'],str(sum(st.session_state.pages))),
                     (t['details_detect'],'Textual / heuristic'),(t['details_ai'],'Ollama · optional')]
        st.markdown('<div class="dl-card">'+''.join(f'<div class="dl-editorial-row"><span>{safe(label)}</span><b>{safe(value)}</b></div>' for label,value in detail_rows)+'</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="dl-note">{safe(t["details_note"])}</div>',unsafe_allow_html=True)
    section('04 / EXPORT',t['generate_title'],t['generate_desc'],tight=True)
    if st.button(t['generate'],disabled=not bool(llm)):
        try:
            selected=st.session_state.changes[:40]
            text='\n'.join(f"CHANGE {i+1} | TYPE={c['type']} | BEFORE={(c['old'].text[:450] if c['old'] else '-')} | AFTER={(c['new'].text[:450] if c['new'] else '-')}" for i,c in enumerate(selected))
            source=type('Source',(),{'doc':'Detected changes / Diferencias detectadas','page':1,'text':text})()
            prompt=('Write a concise executive summary IN ENGLISH, organized by topic. Do not conflate SLA credits with termination penalties, or audits with incident notices. Do not invent findings. Cite [SOURCE 1].' if lang=='en' else
                    'Escribe un resumen ejecutivo conciso EN ESPAÑOL, agrupado por temas. No confundas créditos SLA con penalizaciones por terminación ni auditorías con notificación de incidentes. No inventes hallazgos. Cita [FUENTE 1].')
            with st.spinner(t['generating']):
                st.session_state.analysis=ask_ollama(prompt,[(source,1.)],llm,language=lang)
        except Exception as exc:st.error(f"{t['error_ai']} {exc}")
    if st.session_state.get('analysis'):
        st.markdown('### '+t['generated']);st.markdown(st.session_state.analysis)
    pdf=make_report(pair[1],pair[3],st.session_state.changes,
                    st.session_state.get('analysis',''),language=lang)
    st.download_button(t['download'],data=pdf,file_name=f'DocuLens_Report_{lang.upper()}.pdf',
                       mime='application/pdf',type='primary')

st.markdown(f'<footer class="dl-footer">DOCULENS / DIMENSION EDITION v0.5 · {safe(t["footer"])}</footer>',
            unsafe_allow_html=True)
