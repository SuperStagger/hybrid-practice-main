"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).
FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle.

{ir_spec}

Guidelines:
1. Identify the main entities and define integer variables for them.
2. NEVER use strings in variable domains. For categorical values (like drinks, slots, days), secretly map them to integers (e.g., 1, 2, 3) and use these integers in your constraints.
3. For a queue/line where 1 is the VERY FRONT and N is the VERY BACK, use domain [1, N].
4. Put ALL distinct variables into a SINGLE list inside `all_different`: [["Var1", "Var2", "Var3"]].
5. "A is somewhere in front of / before B" means A < B.
6. "A is directly behind B" means A == B + 1. "A is directly in front of B" means A + 1 == B.
7. "Not at the back" means A != N. "Not at the front" means A != 1.

Example Puzzle:
Three people (Alice, Bob, Charlie) stand in a queue from 1 (front) to 3 (back).
Alice is somewhere in front of Charlie.
Bob is directly behind Alice.
Charlie is not at the back.

Example IR:
{{
  "variables": {{
    "Alice": [1, 3],
    "Bob": [1, 3],
    "Charlie": [1, 3]
  }},
  "all_different": [
    ["Alice", "Bob", "Charlie"]
  ],
  "constraints": [
    "Alice < Charlie",
    "Bob == Alice + 1",
    "Charlie != 3"
  ]
}}

Puzzle:
{text}
{feedback}"""


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid."""
    errors = []
    
    # Перевірка правильності доменів змінних
    variables = ir.get("variables", {})
    for name, domain in variables.items():
        if isinstance(domain, list) and len(domain) == 2:
            if domain[0] > domain[1]:
                errors.append(f"Invalid domain for '{name}': min ({domain[0]}) cannot be greater than max ({domain[1]}).")
        elif isinstance(domain, dict) and "values" in domain:
            if not all(isinstance(v, int) for v in domain["values"]):
                errors.append(f"Invalid values for '{name}': all values must be integers.")
        else:
            errors.append(f"Invalid domain format for '{name}'. Must be [lo, hi] or {{'values': [...]}}.")

    # Перевірка синтаксису та наявності неіснуючих змінних за допомогою ir_solver
    try:
        ir_solver.build(ir)
    except ValueError as e:
        errors.append(f"Syntax/Solver error: {e}")
    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return errors


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}."""
    return ir_solver.solve(ir, limit=2)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format.

    Provided baseline: the model translates the assignment. This is a weak point - the model can
    introduce errors AFTER the solver. Status and objective are always taken from the solver, never
    from the model. Extension: write deterministic code for problem types where the mapping is obvious.
    """
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
    prompt = PRESENT_PROMPT.format(text=problem["text"], ir=json.dumps(ir, ensure_ascii=False),
                                   assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
                                   answer_format=problem["answer_format"])
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    # TODO(team): if `errors` is not empty, make ONE more attempt:
    #   ir = formalize(problem, feedback="\n".join(errors)); errors = validate_ir(ir); record it in trace.
    if errors:
        feedback_msg = "\n".join(errors)
        ir = formalize(problem, feedback=feedback_msg)
        errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": errors})
        
        # Якщо після другої спроби все ще є помилки, тоді зупиняємось
        if errors:
            raise HybridError("invalid IR after retry: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:                      # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {"status": result["status"], "n_solutions": len(result["solutions"]),
                       "objective": result.get("objective"),
                       "first_solution": result["solutions"][0] if result["solutions"] else None}

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
