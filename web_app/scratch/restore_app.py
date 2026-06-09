import json
from pathlib import Path

log_path = Path(r"C:\Users\carlo\.gemini\antigravity\brain\e640c8d4-4347-4452-b18d-203592c70309\.system_generated\logs\transcript.jsonl")
if not log_path.exists():
    print(f"Log file not found at {log_path}")
    exit(1)

print("Reading transcript...")
with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Found {len(lines)} lines in log.")

# Iterate backward to find the most recent complete app.py or large write
for i, line in enumerate(reversed(lines)):
    try:
        step = json.loads(line)
        tool_calls = step.get("tool_calls", [])
        for tc in tool_calls:
            func_name = tc.get("name")
            args = tc.get("args", {})
            if "TargetFile" in args and "app.py" in args["TargetFile"]:
                print(f"Found tool call: {func_name} at index {len(lines) - 1 - i}")
                # Print keys or summary of args
                print(f"Args keys: {list(args.keys())}")
                if "CodeContent" in args:
                    print("Found CodeContent!")
                    # Save it!
                    out_path = Path("scratch/restored_app.py")
                    out_path.parent.mkdir(exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as out:
                        out.write(args["CodeContent"])
                    print(f"Saved restored code to {out_path}")
                    break
    except Exception as e:
        continue
