"""Runner refactor contracts: moved factories stay importable, CLI unchanged."""

import subprocess
import sys



def test_runners_import_in_either_order_without_a_cycle():
    for first, second in (("runners.run_task3", "runners.run_task"), ("runners.run_task", "runners.run_task3")):
        code = f"import {first}, {second}"
        subprocess.run([sys.executable, "-c", code], check=True, cwd="src")
