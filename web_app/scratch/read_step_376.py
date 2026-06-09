import json
from pathlib import Path

log_path = Path(r"C:\Users\carlo\.gemini\antigravity\brain\e640c8d4-4347-4452-b18d-203592c70309\.system_generated\logs\transcript.jsonl")
if not log_path.exists():
    print("Log not found")
    exit(1)

with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# We saw it was step index 376 in the line list, let's look around index 376
for idx in range(370, 385):
    if idx >= len(lines):
        continue
    try:
        step = json.loads(lines[idx])
        step_idx = step.get("step_index")
        for tc in step.get("tool_calls", []):
            args = tc.get("args", {})
            if "TargetFile" in args and "app.py" in args["TargetFile"]:
                print(f"Log Index {idx} (Step {step_idx}): {tc['name']}")
                if "ReplacementContent" in args:
                    content = args["ReplacementContent"]
                    print(f"Content length: {len(content)}")
                    print("Is truncated marker present?", "truncated" in content or "..." in content[-20:])
                    # Write to file
                    out_path = Path(f"scratch/step_{step_idx}_content.py")
                    out_path.write_text(content, encoding="utf-8")
                    print(f"Saved content to {out_path}")
    except Exception as e:
        print(f"Error at {idx}: {e}")
