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
            step_index = r.get("step_index")
            tool_calls = r.get("tool_calls", [])
            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args", {})
                target = args.get("TargetFile") or args.get("AbsolutePath") or ""
                if "app.py" in str(target):
                    # Check size of replacement
                    content = args.get("ReplacementContent") or args.get("CodeContent") or ""
                    if len(content) > 5000:
                        print(f"Step {step_index}: {name} on {target}, content length = {len(content)}")
                        # Check if it has any truncation marker
                        print("  Is truncated?", "truncated" in content or "..." in content[-50:])
                    elif name == "multi_replace_file_content":
                        chunks = args.get("ReplacementChunks", [])
                        if isinstance(chunks, str):
                            try: chunks = json.loads(chunks)
                            except: pass
                        total_size = sum(len(str(c.get("ReplacementContent", ""))) for c in chunks if isinstance(c, dict))
                        if total_size > 5000:
                            print(f"Step {step_index}: multi_replace_file_content on {target}, total replacement size = {total_size}")
        except Exception as e:
            pass
