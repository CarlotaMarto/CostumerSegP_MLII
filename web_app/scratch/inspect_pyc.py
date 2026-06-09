import marshal
from pathlib import Path

pyc_path = Path("__pycache__/app.cpython-312.pyc")
if not pyc_path.exists():
    print("pyc not found!")
    exit(1)

with open(pyc_path, "rb") as f:
    f.read(16)
    code_obj = marshal.load(f)

out_file = Path("scratch/pyc_constants.txt")

with open(out_file, "w", encoding="utf-8") as out:
    def dump_strings(code, depth=0):
        indent = "  " * depth
        for c in code.co_consts:
            if isinstance(c, str):
                out.write(f"{indent}String: {repr(c)}\n")
            elif type(c).__name__ == "code":
                out.write(f"{indent}Sub-code: {code.co_name} -> {c.co_name}\n")
                dump_strings(c, depth + 1)
    dump_strings(code_obj)

print(f"Dumped code object constants to {out_file}")
