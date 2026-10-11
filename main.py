import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Let any website call your API (the grader needs this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Shapes of data ----------
class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------- Part 1: The kitchen ----------
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()  # catch everything print() says
    try:
        exec(code, {"__name__": "__main__"})  # run the code in its own clean "room"
        return {"success": True, "output": sys.stdout.getvalue()}
    except Exception as e:
        # Skip our own helper's frame so the traceback looks like the example
        tb = e.__traceback__.tb_next if e.__traceback__ else None
        output = "".join(traceback.format_exception(type(e), e, tb))
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout  # always put the real screen back


# ---------- Part 2: The head chef (AI) ----------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Number the lines so the AI doesn't have to count
    numbered = "\n".join(
        f"{i}: {line}" for i, line in enumerate(code.split("\n"), start=1)
    )

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.
Line numbers start at 1 and are shown at the start of each line.

CODE:
{numbered}

TRACEBACK:
{tb}

Return the line number(s) where the error is located.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ErrorAnalysis,
        ),
    )
    return ErrorAnalysis.model_validate_json(response.text).error_lines


# Backup plan if the AI call fails: read the line number from the traceback
def fallback_lines(tb: str) -> List[int]:
    nums = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(nums[-1])] if nums else []


# ---------- The waiter ----------
@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    run = execute_python_code(req.code)

    if run["success"]:
        return {"error": [], "result": run["output"]}

    try:
        lines = analyze_error_with_ai(req.code, run["output"])
    except Exception:
        lines = fallback_lines(run["output"])

    return {"error": lines, "result": run["output"]}