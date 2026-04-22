from __future__ import annotations

import os

import streamlit as st

from src.llm_backend import load_text_generation_backend
from src.paths import DATA_RAW_DIR, UNCAPOPT_PATH, list_instance_files
from src.pipeline_trace import PipelineTrace
from src.poc_pipeline import run_poc_scenario


def render_trace(trace: PipelineTrace) -> None:
    st.subheader("Pipeline execution trace")
    for idx, step in enumerate(trace.steps, start=1):
        label = f"{idx}. {step.name} - {step.status}"
        if step.duration_s is not None:
            label += f" ({step.duration_s:.3f}s)"
        with st.expander(label, expanded=(idx == 1)):
            if step.error:
                st.error(step.error)

            for key, value in step.artifacts.items():
                if key == "generated_code":
                    st.markdown("**generated_code**")
                    st.code(value, language="python")
                elif key in ("system_prompt", "user_prompt", "raw_llm_output"):
                    st.markdown(f"**{key}**")
                    st.code(str(value))
                else:
                    st.write({key: value})


def _backend_info_line() -> str:
    return load_text_generation_backend().describe()


def main() -> None:
    st.title("CSC5382 - Milestone 2 PoC: UFLP baseline with optional LLM verification")
    st.caption(
        "The deterministic CBC baseline always runs first. Optionally, an off-the-shelf LLM can generate "
        "solver code that is executed in a sandbox and checked against the baseline objective."
    )

    instances = list_instance_files()
    if not instances:
        st.error(f"No instances found in: {DATA_RAW_DIR}")
        st.stop()

    instance_path = st.selectbox("Choose instance", instances, format_func=lambda path: path.name)
    optfile_path = str(UNCAPOPT_PATH) if UNCAPOPT_PATH.exists() else None

    if not optfile_path:
        st.warning("uncapopt.txt not found - baseline will run without best-known gap computation.")

    default_enable_llm = bool((os.getenv("GROQ_API_KEY") or "").strip())
    enable_llm = st.checkbox(
        "Enable LLM-generated solver verification",
        value=default_enable_llm,
        help=(
            "When enabled, the app asks the configured LLM backend to generate solver code for the same "
            "instance and then compares that generated solver output against the deterministic baseline."
        ),
    )

    if enable_llm:
        st.info(_backend_info_line())

    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_instance" not in st.session_state:
        st.session_state.last_instance = None
    if "last_enable_llm" not in st.session_state:
        st.session_state.last_enable_llm = None

    run_btn = st.button("Run", type="primary")

    if run_btn:
        trace = PipelineTrace()
        st.session_state.last_trace = trace
        st.session_state.last_result = None
        st.session_state.last_instance = str(instance_path)
        st.session_state.last_enable_llm = enable_llm

        status_box = st.status("Running PoC pipeline...", expanded=True)

        try:
            status_box.update(
                label="Running deterministic baseline and optional LLM verification...",
                state="running",
            )
            result = run_poc_scenario(
                str(instance_path),
                optfile_path,
                enable_llm=enable_llm,
                trace=trace,
            )
            st.session_state.last_result = result
            status_box.update(label="Done.", state="complete")
        except Exception as exc:
            status_box.update(label=f"Failed: {type(exc).__name__}", state="error")
            st.error(f"{type(exc).__name__}: {exc}")

    if st.session_state.last_trace is None:
        return

    instance_name = os.path.basename(st.session_state.last_instance)
    llm_state = "enabled" if st.session_state.last_enable_llm else "disabled"
    st.divider()
    st.caption(f"Last run: baseline scenario on **{instance_name}** with LLM verification **{llm_state}**")

    result = st.session_state.last_result
    st.subheader("Results summary")

    if result is None:
        st.info("No results produced (pipeline likely failed early).")
        render_trace(st.session_state.last_trace)
        return

    baseline = result.baseline
    st.metric("Baseline objective", f"{baseline.objective:.3f}")
    if baseline.gap_percent is not None:
        st.metric("Baseline gap vs best known (%)", f"{baseline.gap_percent:.6f}")
    st.write(f"Open facilities ({len(baseline.open_facilities)}):", baseline.open_facilities)

    llm = result.llm
    if llm.status == "SKIPPED":
        st.info("LLM verification was skipped for this run.")
    elif llm.status == "OK":
        st.success(f"LLM solver succeeded ({llm.backend_name}/{llm.model_name})")
        st.metric("LLM objective", f"{llm.objective:.3f}")
        st.metric("LLM gap vs baseline (%)", f"{llm.gap_vs_baseline_pct:.6f}")
        st.write(f"LLM open facilities ({len(llm.open_facilities)}):", llm.open_facilities)
        if llm.assignments:
            st.write("Assignments preview (first 20 customers):", llm.assignments[:20])
    else:
        st.error(f"LLM solver failed ({llm.backend_name}/{llm.model_name})")
        if llm.error:
            st.code(llm.error)

    render_trace(st.session_state.last_trace)


if __name__ == "__main__":
    main()
