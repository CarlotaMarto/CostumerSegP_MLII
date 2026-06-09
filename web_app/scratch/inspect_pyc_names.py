import marshal
from pathlib import Path

pyc_path = Path("__pycache__/app.cpython-312.pyc")
if not pyc_path.exists():
    print("pyc not found!")
    exit(1)

with open(pyc_path, "rb") as f:
    f.read(16)
    code_obj = marshal.load(f)

out_file = Path("scratch/pyc_names.txt")

with open(out_file, "w", encoding="utf-8") as out:
    def dump_names(code, depth=0):
        indent = "  " * depth
        out.write(f"{indent}Code block: {code.co_name}\n")
        out.write(f"{indent}  Names: {list(code.co_names)}\n")
        out.write(f"{indent}  Varnames: {list(code.co_varnames)}\n")
        for c in code.co_consts:
            if type(c).__name__ == "code":
                dump_names(c, depth + 1)
                
    dump_names(code_obj)

print(f"Dumped code names to {out_file}")
