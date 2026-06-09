import json
from pathlib import Path

transcript_path = Path(r"C:\Users\carlo\.gemini\antigravity\brain\e640c8d4-4347-4452-b18d-203592c70309\.system_generated\logs\transcript.jsonl")

if not transcript_path.exists():
    print("Transcript not found!")
    exit(1)

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get("step_index") == 306:
                print("Found Step 306!")
                tool_calls = r.get("tool_calls", [])
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("args", {})
                    target = args.get("TargetFile") or ""
                    print(f"Tool: {name}, Target: {target}")
                    if "app.py" in target and "CodeContent" in args:
                        code = args["CodeContent"]
                        print(f"Code size: {len(code)}")
                        out_path = Path("scratch/recovered_step_306_app.py")
                        out_path.write_text(code, encoding="utf-8")
                        print(f"Saved recovered code to {out_path}")
        except Exception as e:
            pass
