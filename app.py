
import streamlit as st

from research import (
    run_research, save_memory, get_memory, get_last_report, clear_memory
)

st.set_page_config(page_title="Agentic Research Analyst", layout="wide")
st.title("Agentic Research Analyst")
st.caption("Plan, search, extract, critique, re-search, report. All on a local model.")


if "last_report" not in st.session_state:
    st.session_state.last_report = get_last_report()   # restored from memory.json
if "result" not in st.session_state:
    st.session_state.result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


with st.sidebar:
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
    st.info("Previous report loaded. This run will build on it as a follow-up.")
else:
    st.caption("No previous context. The first query starts fresh.")


query = st.text_input(
    "Research question",
    placeholder="Use at least 4 words, e.g. 'NVIDIA Blackwell GPU data center 2024'",
)

if st.button("Run Research", type="primary"):
    if len(query.split()) < 4:
        st.error("Too vague. Use at least 4 words, e.g. 'NVIDIA Blackwell GPU data center 2024'.")
    else:
        status = st.status("Running research...", expanded=True)
        try:
            result = run_research(
                query,
                previous_report=st.session_state.last_report,
                progress=lambda msg: status.write(msg),
            )
            status.update(label="Research complete", state="complete", expanded=False)

            save_memory(query, result["report"])
            st.session_state.last_report = result["report"]
            st.session_state.last_query = query
            st.session_state.result = result
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong: {e}")
            st.caption("Check that the Ollama app is running and the model name in research.py is correct.")

result = st.session_state.result
if result:
    with st.expander("1. Research plan"):
        st.write(result["plan"])

    with st.expander("2. Extracted facts"):
        st.write(result["facts"])

    with st.expander("3. Critique"):
        st.write(result["critique"])

    with st.expander("4. Round 2 follow-up"):
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