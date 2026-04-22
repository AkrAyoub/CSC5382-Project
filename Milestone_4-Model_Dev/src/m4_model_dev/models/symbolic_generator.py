from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from m4_model_dev.models.model_registry import CandidateSpec


SYSTEM_PROMPT = (
    "You are an expert operations research engineer. "
    "Return ONLY Python code between the required markers and do not add commentary."
)

TOKEN_V0_PROMPT = """
Write ONE self-contained Python module that solves the OR-Library UNCAPACITATED Facility Location Problem (UFLP) with OR-Tools CBC.

ABSOLUTE RULES:
- Define solve(instance_path: str) -> dict with keys objective, open_facilities, assignments
- Imports allowed ONLY: pathlib, typing, ortools.linear_solver.pywraplp
- No prints, no logging, no filesystem writes
- Parse the file with token-stream parsing using Path(instance_path).read_text().split()
- Build a CBC MILP with BoolVar y[i] and x[j][i]
- Required parse shape:
  1. tokens = Path(instance_path).read_text().split()
  2. header = [float(tokens[0]), float(tokens[1])]
  3. m = int(header[0]); n = int(header[1]); cursor = 2
  4. read m facility lines and keep only the fixed cost value
  5. for each of the n customers, read one demand token, then accumulate assignment costs until len(row) == m
  6. build assignments as one chosen facility index per customer, not a matrix
- Return a dict with objective, open_facilities, assignments

OUTPUT FORMAT:
===BEGIN_CODE===
# python here
===END_CODE===
""".strip()

ROBUST_V1_PROMPT = """
Write ONE self-contained Python module that solves the OR-Library UNCAPACITATED Facility Location Problem (UFLP) with OR-Tools CBC.

ABSOLUTE RULES:
- Define solve(instance_path: str) -> dict with keys objective, open_facilities, assignments
- Imports allowed ONLY: pathlib, typing, re, ortools.linear_solver.pywraplp
- No prints, no logging, no filesystem writes
- You MUST support both OR-Library file variants:
  1. facility lines with two numeric tokens
  2. facility lines where the capacity token is written literally and only the fixed cost is numeric
- Use line-based parsing with numeric-token extraction from each line
- Required parse shape:
  1. lines = [line.strip() for line in Path(instance_path).read_text().splitlines() if line.strip()]
  2. header = _numbers(lines[0]); m = int(header[0]); n = int(header[1]); line_idx = 1
  3. for _ in range(m): values = _numbers(lines[line_idx]); line_idx += 1; fixed.append(values[-1])
  4. for each customer: ignore the single demand line, then accumulate assignment costs with while len(row) < m
  5. open_facilities must be integer facility ids only
  6. assignments must be one integer facility id per customer, not a matrix
- Build a CBC MILP with BoolVar y[i] and x[j][i]
- Return a dict with objective, open_facilities, assignments
- Add basic post-solve validation for assignments

OUTPUT FORMAT:
===BEGIN_CODE===
# python here
===END_CODE===
""".strip()


@dataclass(frozen=True)
class GeneratedCodeResult:
    code: str
    raw_text: str
    backend_name: str
    model_name: str
    prompt_template: str


class GenValidationError(RuntimeError):
    def __init__(self, code: str, message: str, raw_preview: str | None = None):
        full = f"{code}: {message}"
        if raw_preview:
            full += f"\n\nRAW_PREVIEW:\n{raw_preview}"
        super().__init__(full)
        self.code = code
        self.raw_preview = raw_preview or ""


def _parse_retry_after_seconds(err_text: str) -> float | None:
    match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", err_text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _extract_code_block(text: str) -> str:
    begin = "===BEGIN_CODE==="
    end = "===END_CODE==="
    start_idx = text.find(begin)
    end_idx = text.find(end)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        code = text[start_idx + len(begin) : end_idx].strip()
        if code:
            return code

    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        code = (match.group(1) or "").strip()
        if code:
            return code

    if "def solve(" in text:
        return text.strip()
    raise RuntimeError("LLM did not return extractable code.")


def _prompt_for_template(prompt_template: str) -> str:
    if prompt_template == "token_v0":
        return TOKEN_V0_PROMPT
    if prompt_template == "robust_v1":
        return ROBUST_V1_PROMPT
    raise ValueError(f"Unsupported prompt template: {prompt_template}")


def _repair_trivial(code: str) -> str:
    code = code.replace("solver.Sum(", "sum(").replace("solver.sum(", "sum(")
    code = re.sub(r"CreateSolver\(\s*['\"]cbc['\"]\s*\)", 'CreateSolver("CBC")', code)
    code = re.sub(r"CreateSolver\(\s*['\"]Cbc['\"]\s*\)", 'CreateSolver("CBC")', code)
    return code


def _validate_common(code: str, raw_text: str) -> None:
    lowered = code.lower()
    banned = [
        "import os",
        "import sys",
        "import subprocess",
        "from os",
        "from sys",
        "eval(",
        "exec(",
        "numpy",
        "pandas",
        "print(",
    ]
    for snippet in banned:
        if snippet in lowered:
            raise GenValidationError("BANNED_SNIPPET", f"Contains banned snippet: {snippet}", raw_preview=raw_text[:1200])

    if "def solve(" not in code:
        raise GenValidationError("MISSING_SOLVE_DEF", "Must define solve(instance_path: str).", raw_preview=raw_text[:1200])
    if 'CreateSolver("CBC")' not in code and "CreateSolver('CBC')" not in code:
        raise GenValidationError("MISSING_CBC", "Must create the solver with CreateSolver('CBC').", raw_preview=raw_text[:1200])
    if not all(token in lowered for token in ["objective", "open_facilities", "assignments"]):
        raise GenValidationError("MISSING_RETURN_KEYS", "Return dict must include objective/open_facilities/assignments.", raw_preview=raw_text[:1200])


def _validate_token_v0(code: str, raw_text: str) -> None:
    _validate_common(code, raw_text)
    lowered = code.lower()
    compact = lowered.replace(" ", "")
    if "read_text().split()" not in compact and "read_text(encoding=" not in compact:
        raise GenValidationError("MISSING_TOKEN_PARSE", "token_v0 must use Path(instance_path).read_text().split().", raw_preview=raw_text[:1200])
    if "header" not in compact or "m=int(header[0])" not in compact or "n=int(header[1])" not in compact:
        raise GenValidationError("MISSING_HEADER_PARSE", "token_v0 must parse m and n from the header tokens.", raw_preview=raw_text[:1200])
    if not re.search(r"for[a-z0-9_]+inrange\(m\)", compact):
        raise GenValidationError("MISSING_FACILITY_LOOP", "token_v0 must iterate over facilities with range(m).", raw_preview=raw_text[:1200])
    if not re.search(r"for[a-z0-9_]+inrange\(n\)", compact):
        raise GenValidationError("MISSING_CUSTOMER_LOOP", "token_v0 must iterate over customers with range(n).", raw_preview=raw_text[:1200])
    if "whilelen(row)<m" not in compact:
        raise GenValidationError("MISSING_COST_ACCUMULATION", "token_v0 must accumulate customer costs until len(row) == m.", raw_preview=raw_text[:1200])
    if "assignments.append(chosen)" not in compact and "assignments[j]=chosen" not in compact:
        raise GenValidationError("INVALID_ASSIGNMENTS_SHAPE", "token_v0 must build assignments as one facility index per customer.", raw_preview=raw_text[:1200])


def _validate_robust_v1(code: str, raw_text: str) -> None:
    _validate_common(code, raw_text)
    lowered = code.lower()
    if "splitlines" not in lowered:
        raise GenValidationError("MISSING_LINE_PARSE", "robust_v1 must parse with splitlines().", raw_preview=raw_text[:1200])
    if "re" not in lowered and "number_pattern" not in lowered:
        raise GenValidationError("MISSING_REGEX_PARSE", "robust_v1 must extract numeric tokens with regex.", raw_preview=raw_text[:1200])
    compact = lowered.replace(" ", "")
    if "header" not in compact or "m=int(header[0])" not in compact or "n=int(header[1])" not in compact:
        raise GenValidationError("MISSING_HEADER_PARSE", "robust_v1 must parse m and n from the header line.", raw_preview=raw_text[:1200])
    if not re.search(r"for[a-z0-9_]+inrange\(m\)", compact):
        raise GenValidationError("MISSING_FACILITY_LOOP", "robust_v1 must iterate over facilities with range(m).", raw_preview=raw_text[:1200])
    if not re.search(r"for[a-z0-9_]+inrange\(n\)", compact):
        raise GenValidationError("MISSING_CUSTOMER_LOOP", "robust_v1 must iterate over customers with range(n).", raw_preview=raw_text[:1200])
    if "whilelen(row)<m" not in compact:
        raise GenValidationError("MISSING_COST_ACCUMULATION", "robust_v1 must accumulate customer costs until len(row) == m.", raw_preview=raw_text[:1200])
    if "assignments.append(chosen)" not in compact and "assignments[j]=chosen" not in compact:
        raise GenValidationError("INVALID_ASSIGNMENTS_SHAPE", "robust_v1 must build assignments as one facility index per customer.", raw_preview=raw_text[:1200])


def _validate_generated_code(code: str, raw_text: str, prompt_template: str) -> None:
    if prompt_template == "token_v0":
        _validate_token_v0(code, raw_text)
        return
    if prompt_template == "robust_v1":
        _validate_robust_v1(code, raw_text)
        return
    raise ValueError(f"Unsupported prompt template: {prompt_template}")


def _build_feedback(last_error: str) -> str:
    return (
        "Your previous attempt failed static validation or sandbox execution.\n"
        f"Error:\n{last_error}\n\n"
        "Regenerate the full module from scratch and fix the issue exactly.\n"
        "Use the required parser shape from the prompt literally, including header = ..., m = int(header[0]), n = int(header[1]), "
        "facility loop over range(m), customer loop over range(n), and while len(row) < m for costs.\n"
        "Do not add commentary. Return only the corrected code block."
    )


def _generate_text_with_groq(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The groq package is not installed. Install milestone 4 requirements first.") from exc

    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    client = Groq(api_key=api_key, max_retries=0)
    last_error: Exception | None = None
    base_sleep = float(os.getenv("GROQ_RETRY_SLEEP", "1.0"))

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            err_text = str(exc)
            if "429" in err_text or "rate limit" in err_text.lower():
                wait_s = _parse_retry_after_seconds(err_text) or (base_sleep * 2.0)
                time.sleep(wait_s + 0.35)
                continue
            if attempt < 3:
                time.sleep(min(base_sleep * attempt, 4.0))
                continue
            raise RuntimeError(f"Groq call failed after {attempt} attempts: {exc}") from exc

    raise RuntimeError(f"Groq call failed: {last_error}")


def generate_solver_code(spec: CandidateSpec) -> GeneratedCodeResult:
    if spec.kind != "llm":
        raise ValueError(f"generate_solver_code only supports llm candidates. Got: {spec.kind}")
    if not spec.model_name:
        raise RuntimeError(f"Candidate '{spec.name}' does not define a model_name.")
    if not spec.prompt_template:
        raise RuntimeError(f"Candidate '{spec.name}' does not define a prompt_template.")
    if spec.backend != "groq":
        raise RuntimeError(f"Unsupported backend '{spec.backend}' for candidate '{spec.name}'.")

    base_user_prompt = _prompt_for_template(spec.prompt_template)
    last_error: str | None = None

    for attempt in range(1, 5):
        user_prompt = base_user_prompt
        if last_error:
            user_prompt += "\n\n# FIX / REPAIR NOTES\n" + _build_feedback(last_error)

        raw = _generate_text_with_groq(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=spec.model_name,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
        code = _repair_trivial(_extract_code_block(raw))

        try:
            _validate_generated_code(code, raw, spec.prompt_template)
            return GeneratedCodeResult(
                code=code,
                raw_text=raw,
                backend_name=spec.backend or "unknown",
                model_name=spec.model_name,
                prompt_template=spec.prompt_template,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(last_error or f"Unable to generate valid code for candidate '{spec.name}'.")
