

import streamlit as st

from research import (
    MODEL, EMBED_MODEL, ResearchError, run_research, validate_query, check_ollama,
    save_memory, save_report_file, get_memory, get_last_report, clear_memory,
)

st.set_page_config(page_title="Agentic Research Analyst", layout="wide")
st.title("Agentic Research Analyst")
st.caption("Plan, search, retrieve, extract, critique, re-search, report. All on a local model.")

st.session_state.setdefault("last_report", get_last_report())
st.session_state.setdefault("result", None)

# system status and memory
with st.sidebar:
    st.header("System")
    try:
        has_embed = check_ollama()
        st.success(f"Ollama running, model {MODEL} ready")
        if has_embed:
            st.success(f"RAG on ({EMBED_MODEL})")
        else:
            st.warning(f"RAG off. Run: ollama pull {EMBED_MODEL}")
    except ResearchError as e:
        st.error(str(e))

    st.header("Memory")
    history = get_memory()
    if history:
        st.write(f"{len(history)} saved report(s)")
        for entry in reversed(history):
            with st.expander(f"{entry['timestamp']}  |  {entry['query'][:40]}"):
                st.write(entry["report"][:600] + "...")
        if st.button("Clear memory"):
            clear_memory()
            st.session_state.last_report = ""
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("Nothing saved yet.")

if st.session_state.last_report:
    st.info("Previous report loaded. The next run will build on it as a follow-up.")
else:
    st.caption("No previous context. The first query starts fresh.")

query = st.text_input(
    "Research question",
    placeholder="NVIDIA Blackwell GPU data center deployment 2024",
)

if st.button("Run Research", type="primary"):
    ok, reason = validate_query(query)
    if not ok:
        st.error(reason)
    else:
        status = st.status("Running research...", expanded=True)
        try:
            result = run_research(
                query,
                previous_report=st.session_state.last_report,
                progress=lambda msg: status.write(msg),
            )
            status.update(label=f"Research complete in {result['seconds']}s",
                          state="complete", expanded=False)
            save_memory(query, result["report"])
            save_report_file(query, result["report"])
            st.session_state.last_report = result["report"]
            st.session_state.result = result
        except ResearchError as e:
            status.update(label="Could not complete", state="error")
            st.error(str(e))
        except Exception as e:
            status.update(label="Unexpected error", state="error")
            st.error(f"Unexpected error: {e}")

# ---- results (read from session state so they stay visible after any click)
result = st.session_state.result
if result:
    st.caption(f"Retrieval mode: {result['mode']}  |  Pages used: {len(result['sources'])}")

    with st.expander("1. Research plan"):
        st.write(result["plan"])

    with st.expander("2. Retrieved evidence (what the Researcher actually saw)"):
        st.text(result["context"])

    with st.expander("3. Extracted facts"):
        st.write(result["facts"])

    with st.expander("4. Critique"):
        st.write(result["critique"])

    with st.expander("5. Round 2 follow-up"):
        if result["followups"]:
            st.write("Follow-up searches run:")
            for phrase in result["followups"]:
                st.write(f"- {phrase}")
            st.write(result["round2_facts"] or "No new facts found.")
        else:
            st.write("The critic suggested no usable follow-up searches.")

    st.subheader("Final report")
    st.write(result["report"])

    st.download_button(
        "Download report",
        data=result["report"],
        file_name="research_report.txt",
        mime="text/plain",
    )
