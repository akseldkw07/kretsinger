"""
Analyze import dependencies in a Python repo.
Usage: python import_analyzer.py /path/to/repo
"""

from dotenv import load_dotenv
import os
import sys

# Load .env file from kretsinger directory
load_dotenv("/Users/Akseldkw/coding/kretsinger/.env")

# Add to Python path
pythonpath = os.getenv("PYTHONPATH")
if pythonpath:
    for path in pythonpath.split(":"):
        sys.path.insert(0, path)

from kret_utils.constants_kret import KretConstants

"""
Analyze import dependencies in a Python repo.
Usage: python import_analyzer.py /path/to/repo
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
import json
from typing import Set, Dict, List


class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = defaultdict(set)  # {type: set(names)}
        self.local_imports = set()
        self.third_party = set()
        self.stdlib = set()
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name.split(".")[0]
            self.imports["import"].add(module)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module.split(".")[0]
            self.imports["from"].add(module)
        self.generic_visit(node)


def analyze_file(filepath: str) -> ImportAnalyzer:
    """Analyze a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        analyzer = ImportAnalyzer(filepath)
        analyzer.visit(tree)
        return analyzer
    except SyntaxError as e:
        analyzer = ImportAnalyzer(filepath)
        analyzer.errors.append(f"SyntaxError: {e}")
        return analyzer
    except Exception as e:
        analyzer = ImportAnalyzer(filepath)
        analyzer.errors.append(f"Error: {e}")
        return analyzer


def analyze_repo(repo_path: str | Path) -> Dict:
    """Analyze entire repository."""
    repo_path = Path(repo_path)

    results = {
        "files": {},
        "summary": {"total_files": 0, "total_imports": set(), "third_party": set(), "internal": set()},
    }

    # Find all Python files
    for py_file in repo_path.rglob("*.py"):
        # Skip common directories
        if any(x in py_file.parts for x in [".venv", "venv", "__pycache__", ".git", "node_modules"]):
            continue

        analyzer = analyze_file(str(py_file))
        relative_path = py_file.relative_to(repo_path)

        results["files"][str(relative_path)] = {"imports": dict(analyzer.imports), "errors": analyzer.errors}

        results["summary"]["total_files"] += 1
        results["summary"]["total_imports"].update(
            analyzer.imports.get("import", set()) | analyzer.imports.get("from", set())
        )

    # Separate internal vs external
    internal_modules = {p.stem for p in repo_path.glob("*.py")}
    internal_modules.update({p.name for p in repo_path.iterdir() if p.is_dir() and not p.name.startswith(".")})

    for imp in results["summary"]["total_imports"]:
        if imp in internal_modules or imp.startswith("kretsinger"):
            results["summary"]["internal"].add(imp)
        else:
            results["summary"]["third_party"].add(imp)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python import_analyzer.py <repo_path>")
        sys.exit(1)

    repo_path = sys.argv[1]
    results = analyze_repo(repo_path)

    print("=" * 80)
    print(f"REPO ANALYSIS: {repo_path}")
    print("=" * 80)

    print(f"\nTotal Python files: {results['summary']['total_files']}")
    print(f"Total unique imports: {len(results['summary']['total_imports'])}")
    print(f"Internal imports: {len(results['summary']['internal'])}")
    print(f"Third-party imports: {len(results['summary']['third_party'])}")

    print("\n" + "=" * 80)
    print("THIRD-PARTY DEPENDENCIES")
    print("=" * 80)
    for imp in sorted(results["summary"]["third_party"]):
        print(f"  - {imp}")

    print("\n" + "=" * 80)
    print("INTERNAL MODULES")
    print("=" * 80)
    for imp in sorted(results["summary"]["internal"]):
        print(f"  - {imp}")

    # Save detailed results to JSON
    output_file = KretConstants.DATA_DIR / "import_analysis.json"

    # Convert sets to sorted lists for JSON serialization
    # Also fix the file imports which are still sets
    files_dict = {}
    for filepath, data in results["files"].items():
        files_dict[filepath] = {
            "imports": {k: sorted(v) if isinstance(v, set) else v for k, v in data["imports"].items()},
            "errors": data["errors"],
        }

    json_results = {
        "files": files_dict,
        "summary": {
            "total_files": results["summary"]["total_files"],
            "total_imports": sorted(list(results["summary"]["total_imports"])),
            "third_party": sorted(list(results["summary"]["third_party"])),
            "internal": sorted(list(results["summary"]["internal"])),
        },
    }

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
