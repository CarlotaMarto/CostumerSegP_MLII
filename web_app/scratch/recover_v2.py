import json
from pathlib import Path

transcript_path = Path(r"C:\Users\carlo\.gemini\antigravity\brain\e640c8d4-4347-4452-b18d-203592c70309\.system_generated\logs\transcript.jsonl")

if not transcript_path.exists():
    print("Transcript not found!")
    exit(1)

records = []
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except Exception as e:
            pass

print(f"Total steps: {len(records)}")
for r in records:
    step_index = r.get("step_index")
    if step_index < 400:
        continue
    tool_calls = r.get("tool_calls", [])
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        target = args.get("TargetFile") or args.get("AbsolutePath") or ""
        if "app.py" in str(target):
            print(f"Step {step_index}: {name} on {target}")
            if name == "replace_file_content":
                print(f"  StartLine={args.get('StartLine')}, EndLine={args.get('EndLine')}")
                print(f"  TargetContent size={len(str(args.get('TargetContent', '')))}, ReplacementContent size={len(str(args.get('ReplacementContent', '')))}")
            elif name == "multi_replace_file_content":
                chunks = args.get("ReplacementChunks", [])
                if isinstance(chunks, str):
                    try:
                        chunks = json.loads(chunks)
                    except:
                        pass
                print(f"  multi_replace with {len(chunks)} chunks")
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        print(f"    Chunk {i}: StartLine={chunk.get('StartLine')}, EndLine={chunk.get('EndLine')}, TargetContent size={len(str(chunk.get('TargetContent', '')))}, ReplacementContent size={len(str(chunk.get('ReplacementContent', '')))}")
                    else:
                        print(f"    Chunk {i}: type={type(chunk)}")
