"""Runner refactor contracts: moved factories stay importable, CLI unchanged."""

import subprocess
import sys
import threading

import runners.run_task as run_task
import runners.run_task3 as run_task3
import session.factories as factories


def test_moved_factories_are_still_exported_by_the_runner():
    assert run_task.make_pick_state is factories.make_pick_state
    assert run_task.make_task1_perceive is factories.make_task1_perceive
    assert run_task3.make_pick_state is factories.make_pick_state


def test_runners_import_in_either_order_without_a_cycle():
    for first, second in (("runners.run_task3", "runners.run_task"), ("runners.run_task", "runners.run_task3")):
        code = f"import {first}, {second}"
        subprocess.run([sys.executable, "-c", code], check=True, cwd="src")


def test_stop_handler_is_a_no_op_off_the_main_thread():
    results = []
    event = threading.Event()
    thread = threading.Thread(target=lambda: results.append(run_task3.install_stop_handler(event)))
    thread.start()
    thread.join()
    assert results == [False]
