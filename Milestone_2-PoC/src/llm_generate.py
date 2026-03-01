from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

SYSTEM = (
    "You are an expert operations research engineer. "
    "Return ONLY Python code between the markers. No commentary."
)

# Shorter + stricter prompt (reduces tokens + increases compliance)
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

    i = text.find(begin)
    j = text.find(end)
    if i != -1 and j != -1 and j > i:
        block = text[i + len(begin) : j].strip()
        return block if block else None

    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        code = (m.group(1) or "").strip()
        return code if code else None

    if "def solve(" in text:
        return text.strip()

    return None


def _parse_retry_after_seconds(err_text: str) -> Optional[float]:
    m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", err_text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _repair_trivial(code: str) -> str:
    # Fix common slips deterministically
    code = code.replace("solver.Sum(", "sum(").replace("solver.sum(", "sum(")
    code = re.sub(r"\bargmax\b", "best_i", code, flags=re.IGNORECASE)

    code = re.sub(r"CreateSolver\(\s*['\"]cbc['\"]\s*\)", "CreateSolver('CBC')", code)
    code = re.sub(r"CreateSolver\(\s*['\"]Cbc['\"]\s*\)", "CreateSolver('CBC')", code)

    return code


def _static_validate_generated_code(code: str, raw_text: str) -> None:
    lowered = code.lower()

    # --- Security bans ---
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
    for b in banned:
        if b in lowered:
            raise GenValidationError("BANNED_SNIPPET", f"Contains banned snippet: {b}", raw_preview=raw_text[:1200])

    # forbidden token
    if re.search(r"\bargmax\b", lowered):
        raise GenValidationError("FORBIDDEN_ARGMAX", "Token 'argmax' is forbidden.", raw_preview=raw_text[:1200])

    # Must define solve
    if not re.search(r"def\s+solve\s*\(\s*instance_path\s*:\s*str\s*\)", code) and not re.search(
        r"def\s+solve\s*\(\s*instance_path\s*\)", code
    ):
        raise GenValidationError("MISSING_SOLVE_DEF", "Must define solve(instance_path: str).", raw_preview=raw_text[:1200])

    # Must parse tokens via Path(instance_path).read_text().split()
    if not re.search(r"Path\s*\(\s*instance_path\s*\)\s*\.read_text\s*\(\s*\)\s*\.split\s*\(\s*\)", code):
        raise GenValidationError("MISSING_TOKEN_PARSE", "Must parse tokens via Path(instance_path).read_text().split().", raw_preview=raw_text[:1200])

    if not re.search(r"it\s*=\s*iter\s*\(\s*tokens\s*\)", code):
        raise GenValidationError("MISSING_TOKEN_ITER", "Must define: it = iter(tokens).", raw_preview=raw_text[:1200])

    # Must create CBC solver exactly via CreateSolver('CBC')
    if not re.search(r"CreateSolver\(\s*['\"]CBC['\"]\s*\)", code):
        raise GenValidationError("MISSING_CBC_CREATESOLVER", "Must call CreateSolver('CBC').", raw_preview=raw_text[:1200])

    # Explicitly forbid pywraplp.Solver
    if re.search(r"pywraplp\.Solver\s*\(", code):
        raise GenValidationError("FORBIDDEN_SOLVER_CONSTRUCTOR", "Do NOT use pywraplp.Solver(...). Use CreateSolver('CBC').", raw_preview=raw_text[:1200])

    # Vars must be BoolVar
    if "solver.BoolVar" not in code:
        raise GenValidationError("MISSING_BOOLVAR", "Must create BoolVar variables.", raw_preview=raw_text[:1200])

    # Must solve via status = solver.Solve()
    if not re.search(r"status\s*=\s*solver\.Solve\s*\(\s*\)", code):
        raise GenValidationError("MISSING_SOLVE_CALL", "Must call: status = solver.Solve().", raw_preview=raw_text[:1200])

    if "solver.status(" in lowered:
        raise GenValidationError("USED_SOLVER_STATUS", "Must not call solver.status().", raw_preview=raw_text[:1200])

    # Parsing must be in the correct paired form for facilities
    # Require a loop that reads cap then f inside the SAME loop
    if not re.search(r"for\s+i\s+in\s+range\s*\(\s*m\s*\)\s*:\s*.*next\(it\).*next\(it\)", lowered, flags=re.DOTALL):
        raise GenValidationError(
            "BAD_FACILITY_PARSE",
            "Facility parsing must read cap then f inside the same for i in range(m) loop.",
            raw_preview=raw_text[:1200],
        )

    # Must parse demand inside for j in range(n)
    if "for j in range(n):" not in lowered or "demand" not in lowered:
        raise GenValidationError("MISSING_DEMAND_PARSE", "Must parse demand inside for j in range(n) loop.", raw_preview=raw_text[:1200])

    # Must not skip tokens with range(2*m)
    if re.search(r"range\s*\(\s*2\s*\*\s*m\s*\)", code):
        raise GenValidationError("SKIPPED_TOKENS", "Must not skip tokens via range(2*m).", raw_preview=raw_text[:1200])

    # Must not use solver.Sum
    if "solver.sum(" in lowered or "solver.Sum(" in code:
        raise GenValidationError("SOLVER_SUM_USED", "Do NOT use solver.Sum; use Python sum(...).", raw_preview=raw_text[:1200])

    # Must minimize (reject MAXIMIZE)
    if "maximize" in lowered:
        raise GenValidationError("WRONG_DIRECTION", "Objective must be MINIMIZATION (no MAXIMIZE).", raw_preview=raw_text[:1200])

    # Must include setcoefficients for y and x (or equivalent objective construction)
    # Enforce objective construction style: SetCoefficient only
    if "objective.add" in lowered:
        raise GenValidationError(
            "OBJECTIVE_MUST_USE_SETCOEFFICIENT",
            "Do NOT use objective.Add(...). Build objective ONLY via SetCoefficient(y[i], fixed[i]) and SetCoefficient(x[j][i], cost[j][i]).",
            raw_preview=raw_text[:1200],
        )

    if "setcoefficient(y" not in lowered:
        raise GenValidationError("MISSING_FIXED_COEF", "Must set fixed coefficients via obj.SetCoefficient(y[i], fixed[i]).", raw_preview=raw_text[:1200])

    if "setcoefficient(x" not in lowered:
        raise GenValidationError("MISSING_ASSIGN_COEF", "Must set assignment coefficients via obj.SetCoefficient(x[j][i], cost[j][i]).", raw_preview=raw_text[:1200])
    if "setcoefficient(x" not in lowered:
        raise GenValidationError("MISSING_ASSIGN_COEF", "Must set assignment coefficients for x[j][i].", raw_preview=raw_text[:1200])

    # Must return keys
    has_keys = all(k in lowered for k in ["objective", "open_facilities", "assignments"])
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
    max_attempts: Optional[int] = None,
) -> LLMGenerated:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    if not model:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Compare.py retries at a higher level
    if max_attempts is None:
        max_attempts = 2

    base_sleep = float(os.getenv("GROQ_RETRY_SLEEP", "1.0"))
    debug = os.getenv("GROQ_DEBUG", "0").strip() == "1"

    user_prompt = USER_TEMPLATE
    if extra_instructions:
        user_prompt = USER_TEMPLATE + "\n# FIX/REPAIR NOTES:\n" + extra_instructions.strip() + "\n"

    client = Groq(api_key=api_key, max_retries=0)

    non429_attempt = 0
    rate_limit_hits = 0
    last_err: Optional[Exception] = None

    # Hard cap rate-limit loops so it doesn't run forever
    max_rate_limit_hits = int(os.getenv("GROQ_MAX_429", "12"))

    while non429_attempt < max_attempts:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=650,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = resp.choices[0].message.content or ""
            code = _extract_code_block(raw)
            if not code:
                preview = raw[:1200].replace("\n", "\\n")
                raise RuntimeError(f"LLM did not return extractable code. Raw preview: {preview}")

            code = _repair_trivial(code)
            _static_validate_generated_code(code, raw_text=raw)

            return LLMGenerated(
                code=code,
                raw_text=raw,
                model=model,
                system_prompt=SYSTEM,
                user_prompt=user_prompt,
            )

        except Exception as e:
            last_err = e
            err_text = str(e)

            # 429 handling
            if "429" in err_text or "rate limit" in err_text.lower():
                rate_limit_hits += 1
                if rate_limit_hits > max_rate_limit_hits:
                    raise RuntimeError(f"Groq call failed: {e}") from e

                wait_s = _parse_retry_after_seconds(err_text)
                if wait_s is None:
                    wait_s = base_sleep * 2.0

                # Sleep
                wait_s = float(wait_s) + 0.35

                if debug:
                    print(f"[GROQ_DEBUG] 429 hit #{rate_limit_hits}; sleeping {wait_s:.2f}s then retry (no attempt used)")
                time.sleep(wait_s)
                continue

            # Non-429 errors consume attempts
            non429_attempt += 1
            if debug:
                print(f"[GROQ_DEBUG] non-429 attempt {non429_attempt}/{max_attempts} failed: {e}")
            if non429_attempt < max_attempts:
                time.sleep(min(base_sleep * (2 ** (non429_attempt - 1)), 6.0))
                continue

            raise RuntimeError(f"Groq call failed after {max_attempts} attempts: {e}") from e

    raise RuntimeError(f"Groq call failed: {last_err}")