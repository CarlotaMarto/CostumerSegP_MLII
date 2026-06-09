import json
from pathlib import Path

log_path = Path(r"C:\Users\carlo\.gemini\antigravity\brain\e640c8d4-4347-4452-b18d-203592c70309\.system_generated\logs\transcript.jsonl")
if not log_path.exists():
    print("Log not found")
    exit(1)

with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            step = json.loads(line)
            for tc in step.get("tool_calls", []):
                args = tc.get("args", {})
                if "TargetFile" in args and "app.py" in args["TargetFile"]:
                    name = tc["name"]
                    start = args.get("StartLine", "N/A")
                    end = args.get("EndLine", "N/A")
                    inst = args.get("Instruction", "N/A")
                    print(f"Step {i} ({step.get('step_index')}): {name} L{start}-L{end} - {inst[:60]}")
        except Exception as e:
            pass
