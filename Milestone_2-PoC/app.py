from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from src.baseline_solver import run_baseline
from src.compare import compare_baseline_vs_llm
from src.pipeline_trace import PipelineTrace


ROOT = Path(__file__).resolve().parents[1]  # repo_root
DATA_RAW = ROOT / "data" / "raw"


def list_instances():
    items = sorted(DATA_RAW.glob("cap*.txt"))
    for name in ["capa.txt", "capb.txt", "capc.txt"]:
        p = DATA_RAW / name
        if p.exists():
            items.append(p)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in items:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def render_trace(trace: PipelineTrace):
    st.subheader("Pipeline execution trace")
    for idx, step in enumerate(trace.steps, start=1):
        label = f"{idx}. {step.name} — {step.status}"
        if step.duration_s is not None:
            label += f" ({step.duration_s:.3f}s)"
        with st.expander(label, expanded=(idx == 1)):
            if step.error:
                st.error(step.error)

            for k, v in step.artifacts.items():
                if k == "generated_code":
                    st.markdown("**generated_code**")
                    st.code(v, language="python")
                elif k in ("system_prompt", "user_prompt", "raw_llm_output"):
                    st.markdown(f"**{k}**")
                    st.code(str(v))
                else:
                    st.write({k: v})


def _env_info_line():
    model = os.getenv("GROQ_MODEL", "(not set)")
    key = "set" if (os.getenv("GROQ_API_KEY") or "").strip() else "NOT SET"
    return f"GROQ_MODEL={model}  |  GROQ_API_KEY={key}"


def main():
    st.title("CSC5382 — Milestone 2 Proof of Concept: UFLP Baseline vs LLM-Generated Solver")

    instances = list_instances()
    if not instances:
        st.error(f"No instances found in: {DATA_RAW}")
        st.stop()

    inst = st.selectbox("Choose instance", instances, format_func=lambda p: p.name)
    optfile = DATA_RAW / "uncapopt.txt"
    optfile_path: Optional[str] = str(optfile) if optfile.exists() else None

    if not optfile_path:
        st.warning("uncapopt.txt not found — baseline will run without best-known gap computation.")

    mode = st.radio("Mode", ["A) Baseline only", "B) LLM generates solver + verify vs baseline"])
    st.caption("Mode B uses a fixed internal prompt template")

    if mode.startswith("B"):
        st.info(_env_info_line())

    # Persist last run so trace/results remain visible after reruns
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None
    if "last_instance" not in st.session_state:
        st.session_state.last_instance = None
    if "last_gap_pct" not in st.session_state:
        st.session_state.last_gap_pct = None

    run_btn = st.button("Run", type="primary")

    if run_btn:
        trace = PipelineTrace()
        st.session_state.last_trace = trace
        st.session_state.last_result = None
        st.session_state.last_mode = mode
        st.session_state.last_instance = str(inst)

        # A simple “live” status box while running
        status_box = st.status("Running pipeline...", expanded=True)

        try:
            if mode.startswith("A"):
                status_box.update(label="Running baseline...", state="running")
                s = trace.start("A) Run baseline OR-Tools MILP (deterministic)")

                res = run_baseline(str(inst), optfile_path)

                gap_pct = None
                if getattr(res, "best_known", None) is not None:
                    bk = float(res.best_known)
                    if bk != 0.0:
                        gap_pct = (float(res.objective) - bk) / abs(bk) * 100.0

                s.artifacts["baseline_objective"] = res.objective
                s.artifacts["best_known"] = res.best_known
                s.artifacts["gap_pct"] = gap_pct
                s.artifacts["runtime_s"] = res.runtime_s
                s.artifacts["open_facilities"] = res.open_facilities
                s.end_ok()

                # stash both result + computed gap for display
                st.session_state.last_result = res
                st.session_state.last_gap_pct = gap_pct
                status_box.update(label="Done.", state="complete")

            else:
                status_box.update(label="Running baseline + LLM pipeline...", state="running")
                result = compare_baseline_vs_llm(str(inst), optfile_path, trace)
                st.session_state.last_result = result
                status_box.update(label="Done.", state="complete")

        except Exception as e:
            status_box.update(label=f"Failed: {type(e).__name__}", state="error")
            st.error(f"{type(e).__name__}: {e}")

    if st.session_state.last_trace is None:
        return

    st.divider()
    st.caption(f"Last run: **{st.session_state.last_mode}** on **{Path(st.session_state.last_instance).name}**")

    # Mode A output: baseline_solver
    if st.session_state.last_mode and st.session_state.last_mode.startswith("A"):
        res = st.session_state.last_result
        if res is not None:
            st.subheader("Results summary")
            st.metric("Objective", f"{res.objective:.3f}")
            gap_pct = st.session_state.last_gap_pct
            if gap_pct is not None:
                st.metric("Gap vs best known (%)", f"{gap_pct:.6f}")
            opens = getattr(res, "open_facilities", [])
            st.write(f"Open facilities ({len(opens)}):", opens)

        render_trace(st.session_state.last_trace)
        return

    # Mode B output: CompareResult dataclass
    result = st.session_state.last_result
    st.subheader("Results summary")

    if result is None:
        st.info("No results produced (pipeline likely failed early).")
        render_trace(st.session_state.last_trace)
        return

    st.metric("Baseline objective", f"{result.baseline_objective:.3f}")
    if result.baseline_gap_pct is not None:
        st.metric("Baseline gap vs best known (%)", f"{result.baseline_gap_pct:.6f}")

    if result.llm_status == "OK":
        st.success(f"LLM solver succeeded (model: {result.llm_model})")
        st.metric("LLM objective", f"{result.llm_objective:.3f}")
        st.metric("LLM gap vs baseline (%)", f"{result.llm_gap_vs_baseline_pct:.6f}")

        st.write(f"LLM open facilities ({len(result.llm_open_facilities)}):", result.llm_open_facilities)

        # sanity preview
        if result.llm_assignments:
            st.write("Assignments preview (first 20 customers):", result.llm_assignments[:20])
    else:
        st.error(f"LLM solver failed (model: {result.llm_model})")
        if result.llm_error:
            st.code(result.llm_error)

    render_trace(st.session_state.last_trace)


if __name__ == "__main__":
    main()