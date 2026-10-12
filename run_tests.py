"""Run all tests without pytest:  python run_tests.py   (or: python -m pytest -q)"""
import importlib
import inspect
import pathlib
import sys
import tempfile
import time
import traceback

MODULES = ["tests.test_simulator", "tests.test_preprocessing", "tests.test_models", "tests.test_real_data_format", "tests.test_alerts", "tests.test_api"]


def main() -> int:
    sys.path.insert(0, ".")
    passed = failed = 0
    t_all = time.perf_counter()
    for mod in MODULES:
        m = importlib.import_module(mod)
        path = mod.replace(".", "/") + ".py"
        for name in sorted(n for n in dir(m) if n.startswith("test_")):
            t0 = time.perf_counter()
            try:
                fn = getattr(m, name)
                if "tmp_path" in inspect.signature(fn).parameters:      # tiny stand-in for pytest's fixture
                    with tempfile.TemporaryDirectory() as d:
                        fn(pathlib.Path(d))
                else:
                    fn()
                passed += 1
                print(f"{path}::{name} ... PASS ({time.perf_counter() - t0:.2f}s)")
            except Exception:
                failed += 1
                print(f"{path}::{name} ... FAIL")
                traceback.print_exc()
    print("-" * 70)
    print(f"{passed} passed, {failed} failed in {time.perf_counter() - t_all:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
