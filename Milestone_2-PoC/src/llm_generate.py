from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.llm_backend import TextGenerationBackend, load_text_generation_backend

SYSTEM = (
    "You are an expert operations research engineer. "
    "Return ONLY Python code between the markers. No commentary."
)

USER_TEMPLATE = """
Write ONE self-contained Python module that solves OR-Library UNCAPACITATED Facility Location Problem (UFLP) with OR-Tools CBC.

ABSOLUTE RULES (MUST FOLLOW):
- Define: solve(instance_path: str) -> dict with keys: objective, open_facilities, assignments
- Imports allowed ONLY: pathlib, typing, ortools.linear_solver.pywraplp
- NO prints/logging.

PARSING (token stream ONLY; NO shortcuts):
tokens = Path(instance_path).read_text().split()
it = iter(tokens)
m = int(next(it)); n = int(next(it))
fixed = []
for i in range(m):
    cap = float(next(it))   # parse but ignore
    f = float(next(it))
    fixed.append(f)
cost = []
for j in range(n):
    demand = float(next(it))  # parse but ignore
    row = [float(next(it)) for _ in range(m)]
    cost.append(row)

MODEL (CBC MILP):
- MUST create solver EXACTLY:
  solver = pywraplp.Solver.CreateSolver('CBC')
- DO NOT use pywraplp.Solver('name', ...)
- Vars: y[i] BoolVar, x[j][i] BoolVar
- Constraints:
  for j: sum_i x[j][i] == 1
  for j,i: x[j][i] <= y[i]
- Objective (MINIMIZE):
  sum_i fixed[i]*y[i] + sum_{j,i} cost[j][i]*x[j][i]
- Use Python sum(...). DO NOT use solver.Sum(...).

SOLVE:
status = solver.Solve()
if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE): raise RuntimeError
objective_value = solver.Objective().Value()
open_facilities = [i for i in range(m) if y[i].solution_value() > 0.5]
assignments = length n; each j picks i where x[j][i].solution_value() > 0.5

MANDATORY CHECKS (raise RuntimeError):
- len(assignments)==n
- all 0<=a<m
- all assignments[j] in open_facilities
- Do NOT use token "argmax" anywhere.

OUTPUT FORMAT (MUST):
===BEGIN_CODE===
# python here
===END_CODE===
"""


@dataclass
class LLMGenerated:
    code: str
    raw_text: str
    backend_name: str
    model: str
    system_prompt: str
    user_prompt: str


class GenValidationError(RuntimeError):
    def __init__(self, code: str, msg: str, raw_preview: str | None = None):
        full = f"{code}: {msg}"
        if raw_preview:
            full += f"\n\nRAW_PREVIEW:\n{raw_preview}"
        super().__init__(full)
        self.code = code
        self.msg = msg
        self.raw_preview = raw_preview or ""


def _extract_code_block(text: str) -> Optional[str]:
    if not text:
        return None

    begin = "===BEGIN_CODE==="
    end = "===END_CODE==="

    start_idx = text.find(begin)
    end_idx = text.find(end)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        block = text[start_idx + len(begin) : end_idx].strip()
        return block if block else None

    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        code = (match.group(1) or "").strip()
        return code if code else None

    if "def solve(" in text:
        return text.strip()

    return None


def _repair_trivial(code: str) -> str:
    code = code.replace("solver.Sum(", "sum(").replace("solver.sum(", "sum(")
    code = re.sub(r"\bargmax\b", "best_i", code, flags=re.IGNORECASE)
    code = re.sub(r"CreateSolver\(\s*['\"]cbc['\"]\s*\)", "CreateSolver('CBC')", code)
    code = re.sub(r"CreateSolver\(\s*['\"]Cbc['\"]\s*\)", "CreateSolver('CBC')", code)
    return code


def _static_validate_generated_code(code: str, raw_text: str) -> None:
    lowered = code.lower()

    banned = [
        "import os",
        "import sys",
        "subprocess",
        "eval(",
        "exec(",
        "from src",
        "import src",
        "numpy",
        "pandas",
    ]
    for snippet in banned:
        if snippet in lowered:
            raise GenValidationError("BANNED_SNIPPET", f"Contains banned snippet: {snippet}", raw_preview=raw_text[:1200])

    if re.search(r"\bargmax\b", lowered):
        raise GenValidationError("FORBIDDEN_ARGMAX", "Token 'argmax' is forbidden.", raw_preview=raw_text[:1200])

    if not re.search(r"def\s+solve\s*\(\s*instance_path\s*:\s*str\s*\)", code) and not re.search(
        r"def\s+solve\s*\(\s*instance_path\s*\)", code
    ):
        raise GenValidationError("MISSING_SOLVE_DEF", "Must define solve(instance_path: str).", raw_preview=raw_text[:1200])

    if not re.search(r"Path\s*\(\s*instance_path\s*\)\s*\.read_text\s*\(\s*\)\s*\.split\s*\(\s*\)", code):
        raise GenValidationError("MISSING_TOKEN_PARSE", "Must parse tokens via Path(instance_path).read_text().split().", raw_preview=raw_text[:1200])

    if not re.search(r"it\s*=\s*iter\s*\(\s*tokens\s*\)", code):
        raise GenValidationError("MISSING_TOKEN_ITER", "Must define: it = iter(tokens).", raw_preview=raw_text[:1200])

    if not re.search(r"CreateSolver\(\s*['\"]CBC['\"]\s*\)", code):
        raise GenValidationError("MISSING_CBC_CREATESOLVER", "Must call CreateSolver('CBC').", raw_preview=raw_text[:1200])

    if re.search(r"pywraplp\.Solver\s*\(", code):
        raise GenValidationError(
            "FORBIDDEN_SOLVER_CONSTRUCTOR",
            "Do NOT use pywraplp.Solver(...). Use CreateSolver('CBC').",
            raw_preview=raw_text[:1200],
        )

    if "solver.BoolVar" not in code:
        raise GenValidationError("MISSING_BOOLVAR", "Must create BoolVar variables.", raw_preview=raw_text[:1200])

    if not re.search(r"status\s*=\s*solver\.Solve\s*\(\s*\)", code):
        raise GenValidationError("MISSING_SOLVE_CALL", "Must call: status = solver.Solve().", raw_preview=raw_text[:1200])

    if "solver.status(" in lowered:
        raise GenValidationError("USED_SOLVER_STATUS", "Must not call solver.status().", raw_preview=raw_text[:1200])

    if not re.search(r"for\s+i\s+in\s+range\s*\(\s*m\s*\)\s*:\s*.*next\(it\).*next\(it\)", lowered, flags=re.DOTALL):
        raise GenValidationError(
            "BAD_FACILITY_PARSE",
            "Facility parsing must read cap then f inside the same for i in range(m) loop.",
            raw_preview=raw_text[:1200],
        )

    if "for j in range(n):" not in lowered or "demand" not in lowered:
        raise GenValidationError("MISSING_DEMAND_PARSE", "Must parse demand inside for j in range(n) loop.", raw_preview=raw_text[:1200])

    if re.search(r"range\s*\(\s*2\s*\*\s*m\s*\)", code):
        raise GenValidationError("SKIPPED_TOKENS", "Must not skip tokens via range(2*m).", raw_preview=raw_text[:1200])

    if "solver.sum(" in lowered or "solver.Sum(" in code:
        raise GenValidationError("SOLVER_SUM_USED", "Do NOT use solver.Sum; use Python sum(...).", raw_preview=raw_text[:1200])

    if "maximize" in lowered:
        raise GenValidationError("WRONG_DIRECTION", "Objective must be MINIMIZATION (no MAXIMIZE).", raw_preview=raw_text[:1200])

    if "objective.add" in lowered:
        raise GenValidationError(
            "OBJECTIVE_MUST_USE_SETCOEFFICIENT",
            "Do NOT use objective.Add(...). Build objective ONLY via SetCoefficient(y[i], fixed[i]) and SetCoefficient(x[j][i], cost[j][i]).",
            raw_preview=raw_text[:1200],
        )

    if "setcoefficient(y" not in lowered:
        raise GenValidationError("MISSING_FIXED_COEF", "Must set fixed coefficients via obj.SetCoefficient(y[i], fixed[i]).", raw_preview=raw_text[:1200])

    if "setcoefficient(x" not in lowered:
        raise GenValidationError(
            "MISSING_ASSIGN_COEF",
            "Must set assignment coefficients via obj.SetCoefficient(x[j][i], cost[j][i]).",
            raw_preview=raw_text[:1200],
        )

    has_keys = all(key in lowered for key in ["objective", "open_facilities", "assignments"])
    has_return = bool(re.search(r"\breturn\b", lowered))
    if not (has_keys and has_return):
        raise GenValidationError(
            "MISSING_RETURN_KEYS",
            "Must return a dict containing keys: objective, open_facilities, assignments.",
            raw_preview=raw_text[:1200],
        )


def generate_solver_code(
    model: Optional[str] = None,
    temperature: float = 0.0,
    extra_instructions: Optional[str] = None,
    backend: Optional[TextGenerationBackend] = None,
) -> LLMGenerated:
    user_prompt = USER_TEMPLATE
    if extra_instructions:
        user_prompt = USER_TEMPLATE + "\n# FIX/REPAIR NOTES:\n" + extra_instructions.strip() + "\n"

    active_backend = backend or load_text_generation_backend()
    response = active_backend.generate_text(
        SYSTEM,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=650,
    )

    raw = response.text
    code = _extract_code_block(raw)
    if not code:
        preview = raw[:1200].replace("\n", "\\n")
        raise RuntimeError(f"LLM did not return extractable code. Raw preview: {preview}")

    code = _repair_trivial(code)
    _static_validate_generated_code(code, raw_text=raw)

    return LLMGenerated(
        code=code,
        raw_text=raw,
        backend_name=response.backend_name,
        model=response.model_name,
        system_prompt=SYSTEM,
        user_prompt=user_prompt,
    )
