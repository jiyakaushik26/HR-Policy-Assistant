from __future__ import annotations

from html import escape

import streamlit as st

from src.rag import answer_question


st.set_page_config(page_title="HR Policy Assistant", page_icon="📘", layout="wide")

STYLES = """
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        color: #172d3d;
        background:
            radial-gradient(ellipse at 52% -12%, rgba(27, 128, 119, .16), transparent 38%),
            #f3f7f9;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding: 2rem 2rem 1rem;
    }
    .hr-banner {
        position: relative;
        overflow: hidden;
        padding: 2.25rem 2.5rem 2.15rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 22px;
        background: linear-gradient(118deg, #132b40 0%, #174b59 62%, #19756d 100%);
        box-shadow: 0 16px 38px rgba(19, 43, 64, .18);
    }
    .hr-banner:after {
        position: absolute;
        top: -120px;
        right: -45px;
        width: 340px;
        height: 340px;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 50%;
        box-shadow: 0 0 0 35px rgba(255,255,255,.035), 0 0 0 75px rgba(255,255,255,.025);
        content: "";
        pointer-events: none;
    }
    .hr-banner-badge {
        display: inline-block;
        padding: .42rem .78rem;
        margin-bottom: .9rem;
        border: 1px solid rgba(179, 232, 222, .35);
        border-radius: 999px;
        background: rgba(220, 255, 247, .1);
        color: #c7f3e8;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .11em;
    }
    .hr-banner h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(2.3rem, 5vw, 3.35rem);
        font-weight: 760;
        letter-spacing: -.055em;
        line-height: 1.08;
    }
    .hr-banner p {
        margin: .75rem 0 0;
        color: #d1e5e8;
        font-size: 1.08rem;
        letter-spacing: .005em;
    }
    .hr-stat {
        min-height: 82px;
        padding: 1rem 1.15rem;
        border: 1px solid #dbe6e9;
        border-radius: 14px;
        background: rgba(255,255,255,.92);
        box-shadow: 0 5px 16px rgba(24, 53, 70, .045);
    }
    .hr-stat-value {
        color: #183e4b;
        font-size: 1.08rem;
        font-weight: 760;
        letter-spacing: -.02em;
    }
    .hr-stat-label {
        margin-top: .2rem;
        color: #74868f;
        font-size: .76rem;
        font-weight: 650;
        letter-spacing: .045em;
        text-transform: uppercase;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 1.15rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #dce6e9;
        border-radius: 17px;
        background: #fff;
        box-shadow: 0 8px 24px rgba(24, 53, 70, .055);
    }
    .hr-card-title {
        margin: 0 0 .35rem;
        color: #1a3947;
        font-size: 1.16rem;
        font-weight: 760;
        letter-spacing: -.025em;
    }
    .hr-card-subtitle {
        margin: 0 0 1rem;
        color: #71818a;
        font-size: .9rem;
        line-height: 1.55;
    }
    .hr-how-card {
        min-height: 258px;
        padding: .45rem .2rem;
    }
    .hr-how-step {
        display: flex;
        align-items: center;
        gap: .75rem;
        padding: .72rem 0;
        border-bottom: 1px solid #edf1f2;
        color: #294651;
        font-size: .91rem;
        font-weight: 650;
    }
    .hr-how-step:last-of-type { border-bottom: 0; }
    .hr-step-number {
        display: inline-flex;
        flex: 0 0 29px;
        align-items: center;
        justify-content: center;
        width: 29px;
        height: 29px;
        border-radius: 9px;
        background: #e7f3f1;
        color: #167368;
        font-size: .78rem;
        font-weight: 800;
    }
    .hr-how-note {
        padding: .72rem .8rem;
        margin-top: .8rem;
        border-left: 3px solid #55a49a;
        border-radius: 4px 9px 9px 4px;
        background: #f1f8f7;
        color: #60747a;
        font-size: .8rem;
        line-height: 1.5;
    }
    [data-testid="stTextInput"] input {
        min-height: 3.2rem;
        border: 1px solid #d2dfe2;
        border-radius: 11px;
        background: #fbfdfd;
        color: #1d3541;
        font-size: .95rem;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #24867e;
        box-shadow: 0 0 0 3px rgba(36,134,126,.13);
    }
    [data-testid="stButton"] button {
        min-height: 3.15rem;
        border: 1px solid #087d74;
        border-radius: 11px;
        background: linear-gradient(105deg, #087d74, #12685f);
        color: white;
        font-size: .96rem;
        font-weight: 750;
        box-shadow: 0 7px 16px rgba(8, 125, 116, .2);
    }
    [data-testid="stButton"] button:hover {
        border-color: #075f5a;
        background: #075f5a;
        color: white;
        box-shadow: 0 9px 20px rgba(8, 125, 116, .25);
    }
    .hr-result {
        padding: 1.35rem 1.5rem;
        border: 1px solid #dce6e9;
        border-radius: 16px;
        background: #fff;
        box-shadow: 0 7px 22px rgba(24, 53, 70, .05);
    }
    .hr-answer-card {
        padding: 1.65rem 1.8rem;
        border-color: #c9e0dc;
        border-top: 4px solid #178278;
        background: linear-gradient(130deg, #fff 0%, #f2faf8 100%);
        box-shadow: 0 10px 28px rgba(20, 92, 86, .09);
    }
    .hr-result-title {
        margin: 0 0 .9rem;
        color: #1b3d4a;
        font-size: .94rem;
        font-weight: 800;
        letter-spacing: .065em;
    }
    .hr-answer-text {
        color: #233f4a;
        font-size: 1.12rem;
        font-weight: 520;
        line-height: 1.75;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    .hr-source-badge {
        display: inline-flex;
        align-items: center;
        gap: .6rem;
        padding: .6rem .82rem;
        margin: .1rem .4rem .15rem 0;
        border: 1px solid #cfe3df;
        border-radius: 10px;
        background: #f1f8f7;
        color: #214d52;
        font-size: .86rem;
        font-weight: 700;
    }
    .hr-page-tag {
        padding: .28rem .48rem;
        border-radius: 7px;
        background: #dcefeb;
        color: #126e65;
        font-size: .68rem;
        font-weight: 850;
        letter-spacing: .065em;
    }
    .hr-evidence-item {
        padding: .9rem 1rem;
        margin: .55rem 0 0;
        border: 1px solid #e0e9eb;
        border-left: 3px solid #6fa9a2;
        border-radius: 8px;
        background: #f7fafb;
        color: #445b64;
        font-size: .92rem;
        line-height: 1.65;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    .hr-footer {
        padding: 1.6rem .5rem .4rem;
        color: #7c8e94;
        font-size: .79rem;
        letter-spacing: .02em;
        text-align: center;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .8rem; }
        .hr-banner { padding: 1.65rem 1.3rem; }
        .hr-how-card { min-height: auto; }
        .hr-result { padding: 1rem; }
        .hr-answer-card { padding: 1.2rem; }
    }
</style>
"""


def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="hr-banner">
            <div class="hr-banner-badge">RAG POWERED &nbsp;•&nbsp; HR MANUAL</div>
            <h1>HR Policy Assistant</h1>
            <p>Intelligent HR Policy Search • Grounded in your HR Manual</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    stat_columns = st.columns(3)
    stats = [("79 Pages", "Manual length"), ("HR Policy Manual", "Knowledge source"), ("Grounded RAG", "Answer approach")]
    for column, (value, label) in zip(stat_columns, stats):
        with column:
            st.markdown(
                f'<div class="hr-stat"><div class="hr-stat-value">{value}</div>'
                f'<div class="hr-stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")

    main_column, info_column = st.columns([1.75, 1], vertical_alignment="top")
    with main_column:
        with st.container(border=True):
            st.markdown('<div class="hr-card-title">Ask your HR Policy</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="hr-card-subtitle">Search the company manual and get a concise answer with its source.</div>',
                unsafe_allow_html=True,
            )
            question = st.text_input(
                "Question",
                placeholder="e.g., How many days of Privilege Leave do permanent employees receive?",
                label_visibility="collapsed",
            )
            ask_clicked = st.button("Ask Policy Assistant  →", type="primary", use_container_width=True)

    with info_column:
        with st.container(border=True):
            st.markdown('<div class="hr-card-title">How it works</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="hr-how-card">
                    <div class="hr-how-step"><span class="hr-step-number">1</span>Ask a question</div>
                    <div class="hr-how-step"><span class="hr-step-number">2</span>Search the HR Manual</div>
                    <div class="hr-how-step"><span class="hr-step-number">3</span>Get a grounded answer</div>
                    <div class="hr-how-note">Answers are based only on the uploaded HR Policy Manual.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            result = answer_question(question)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.write("")
        st.markdown(
            f'<section class="hr-result hr-answer-card">'
            f'<div class="hr-result-title">💡 &nbsp;ANSWER</div>'
            f'<div class="hr-answer-text">{escape(str(result["answer"]))}</div>'
            f'</section>',
            unsafe_allow_html=True,
        )

        references = result.get("source_references") or []
        if references:
            unique_refs = []
            seen = set()
            for ref in references:
                key = (ref.get("source"), ref.get("page"))
                if key not in seen:
                    seen.add(key)
                    unique_refs.append(ref)
            badges = "".join(
                '<span class="hr-source-badge">'
                f'{escape(str(ref["source"]))}'
                f'<span class="hr-page-tag">PAGE {escape(str(ref["page"]))}</span>'
                '</span>'
                for ref in unique_refs
            )
            st.markdown(
                f'<section class="hr-result"><div class="hr-result-title">📄 &nbsp;SOURCE</div>{badges}</section>',
                unsafe_allow_html=True,
            )

            evidence_items = "".join(
                f'<div class="hr-evidence-item">{escape(str(item["text"]))}</div>'
                for item in result.get("evidence", [])
            )
            st.markdown(
                '<section class="hr-result">'
                '<div class="hr-result-title">🔎 &nbsp;RETRIEVED EVIDENCE</div>'
                f'{evidence_items}</section>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<section class="hr-result"><div class="hr-result-title">📄 &nbsp;SOURCE</div>'
                '<div class="hr-answer-text">Not found</div></section>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<section class="hr-result"><div class="hr-result-title">🔎 &nbsp;RETRIEVED EVIDENCE</div>'
                '<div class="hr-evidence-item">No supporting PDF evidence was retrieved.</div></section>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="hr-footer">HR Policy Assistant • Grounded exclusively in the HR Policy Manual</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
