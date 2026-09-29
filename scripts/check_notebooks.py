"""Validate all notebooks and execute code companions in isolated directories."""

import argparse
import importlib
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    notebooks = sorted(ROOT.glob("*.ipynb"))
    if not notebooks:
        raise RuntimeError("No notebooks found")
    if not args.validate_only:
        # These notebooks catch missing imports; require them so CI cannot skip models.
        for dependency in ("xgboost", "lightgbm", "catboost"):
            importlib.import_module(dependency)
    output_dir = ROOT / "artifacts" / "executed"
    failures = []
    for path in notebooks:
        notebook = None
        execute = False
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            execute = not args.validate_only and any(
                cell.cell_type == "code" and cell.source.strip()
                for cell in notebook.cells
            )
            if execute:
                output_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="ml-notebook-") as workdir:
                    NotebookClient(
                        notebook, timeout=300, kernel_name="python3",
                        allow_errors=False, force_raise_errors=True,
                        resources={"metadata": {"path": workdir}},
                    ).execute()
            print(f"PASS {path.name} ({'executed' if execute else 'validated'})", flush=True)
        except Exception as error:
            failures.append(path.name)
            print(f"FAIL {path.name}: {error}", flush=True)
        finally:
            if execute and notebook is not None:
                nbformat.write(notebook, output_dir / path.name)
    if failures:
        raise SystemExit(f"Failed notebooks: {', '.join(failures)}")
    print(f"Checked {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
