import importlib.util
from pathlib import Path
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t02", here / "02_generate.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

samples = [
    ("\n\n**Abstract**\n\nThe zoonotic origin of ...", "The zoonotic origin of ..."),
    ("*Abstract*\nThe zoonotic origin ...", "The zoonotic origin ..."),
    ("Abstract:\nThe zoonotic origin ...", "The zoonotic origin ..."),
    ("The zoonotic origin of SARS-CoV-2 ...", "The zoonotic origin of SARS-CoV-2 ..."),
    (None, None),
    ("", ""),
]
ok = True
for inp, exp in samples:
    got = m.clean_generated(inp)
    if got != exp:
        ok = False
    print(f"[{'OK' if got==exp else 'FAIL'}] in={repr(inp)[:16]} -> got={repr(got)[:28]} exp={repr(exp)[:28]}")
print("ALL PASS" if ok else "SOME FAILED")
