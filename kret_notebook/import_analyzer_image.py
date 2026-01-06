"""
Analyze import dependencies in a Python repo and generate SVG visualization.
Usage: python import_graph_viz.py /path/to/repo
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
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import json

try:
    import networkx as nx
    import matplotlib.pyplot as plt

    GRAPHVIZ_AVAILABLE = False
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    plt = None
    GRAPHVIZ_AVAILABLE = False
    NETWORKX_AVAILABLE = False
try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    graphviz = None


class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = set()
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name.split(".")[0]
            self.imports.add(module)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module.split(".")[0]
            self.imports.add(module)
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
        "imports": defaultdict(set),
        "summary": {"total_files": 0, "total_imports": set(), "third_party": set(), "internal": set()},
    }

    # Find all Python files
    for py_file in repo_path.rglob("*.py"):
        # Skip common directories
        if any(x in py_file.parts for x in [".venv", "venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]):
            continue

        analyzer = analyze_file(str(py_file))
        relative_path = py_file.relative_to(repo_path)

        results["files"][str(relative_path)] = {"imports": sorted(analyzer.imports), "errors": analyzer.errors}

        # Build import graph
        module_name = py_file.stem
        for imp in analyzer.imports:
            results["imports"][module_name].add(imp)

        results["summary"]["total_files"] += 1
        results["summary"]["total_imports"].update(analyzer.imports)

    # Separate internal vs external
    internal_modules = {p.stem for p in repo_path.glob("*.py")}
    internal_modules.update({p.name for p in repo_path.iterdir() if p.is_dir() and not p.name.startswith(".")})

    for imp in results["summary"]["total_imports"]:
        if imp in internal_modules or any(imp.startswith(f"kret_{x}") for x in ["", "lightning", "utils"]):
            results["summary"]["internal"].add(imp)
        else:
            results["summary"]["third_party"].add(imp)

    return results


def create_graphviz_visualization(results: Dict, output_file: str | Path = "import_dependencies.svg"):
    """Create SVG using Graphviz."""
    if graphviz is None:
        print("ERROR: graphviz not installed. Install with: pip install graphviz")
        return False

    dot = graphviz.Digraph(comment="Import Dependencies", format="svg")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

    # Add nodes for top imports
    all_imports = results["summary"]["total_imports"]
    internal = results["summary"]["internal"]
    third_party = results["summary"]["third_party"]

    # Limit to most common imports to avoid too crowded graph
    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            import_counts[imp] += 1

    # Get top 50 most imported modules
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    top_import_names = {imp for imp, _ in top_imports}

    # Add nodes
    for imp, count in top_imports:
        if imp in internal:
            dot.node(imp, f"{imp}\n({count})", fillcolor="lightgreen")
        else:
            dot.node(imp, f"{imp}\n({count})", fillcolor="lightcoral")

    # Add edges
    edge_count = 0
    for module, imports in results["imports"].items():
        for imp in imports:
            if imp in top_import_names and edge_count < 100:  # Limit edges
                dot.edge(module, imp, weight=str(import_counts[imp]))
                edge_count += 1

    # Save
    output_path = output_file.replace(".svg", "")
    dot.render(output_path, cleanup=True)
    print(f"✓ SVG saved to: {output_file}")
    return True


def create_matplotlib_visualization(results: Dict, output_file: str | Path = "import_dependencies.png"):
    """Create PNG using matplotlib and networkx."""
    if not nx or not plt:
        print("ERROR: networkx/matplotlib not installed. Install with: pip install networkx matplotlib")
        return False

    # Build graph
    G = nx.DiGraph()

    # Count imports
    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            import_counts[imp] += 1

    # Get top 30 imports
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    top_import_names = {imp for imp, _ in top_imports}

    # Add nodes
    internal = results["summary"]["internal"]
    for imp, count in top_imports:
        G.add_node(imp, count=count)

    # Add edges
    edge_count = 0
    for module, imports in results["imports"].items():
        for imp in imports:
            if imp in top_import_names and edge_count < 80:
                G.add_edge(module, imp)
                edge_count += 1

    # Draw
    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Color nodes based on internal/external
    node_colors = []
    for node in G.nodes():
        if node in internal:
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightcoral")

    # Node sizes based on import count
    node_sizes = [import_counts.get(node, 1) * 50 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, width=0.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    plt.title("Python Import Dependency Graph\n(Green = Internal, Red = Third-party)", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ PNG saved to: {output_file}")
    return True


def create_html_interactive(results: Dict, output_file: str | Path = "import_dependencies.html"):
    """Create interactive HTML visualization."""
    try:
        import pyvis.network as net

        assert nx is not None
    except ImportError:
        print("ERROR: pyvis not installed. Install with: pip install pyvis")
        return False
    except AssertionError:
        print("ERROR: networkx not installed. Install with: pip install networkx")
        return False

    # Build graph
    G = nx.DiGraph()

    # Count imports
    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            import_counts[imp] += 1

    # Get top imports
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:40]
    top_import_names = {imp for imp, _ in top_imports}

    internal = results["summary"]["internal"]

    # Add nodes
    for imp, count in top_imports:
        color = "#90EE90" if imp in internal else "#FFB6C6"  # Green for internal, pink for external
        G.add_node(imp, title=f"{imp} (imported {count} times)", color=color, size=min(count * 3, 50))

    # Add edges
    for module, imports in results["imports"].items():
        for imp in imports:
            if imp in top_import_names:
                G.add_edge(module, imp)

    # Create pyvis network
    net_graph = net.Network(height="750px", width="100%", directed=True, physics=True)
    net_graph.from_nx(G)

    # Customize physics
    net_graph.physics(enabled=True, barnesHut={"gravitationalConstant": -10000, "centralGravity": 0.1})

    # Save
    net_graph.show(str(output_file))
    print(f"✓ Interactive HTML saved to: {output_file}")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python import_graph_viz.py <repo_path> [output_format]")
        print("  output_format: graphviz (svg), matplotlib (png), or interactive (html)")
        print("  Default: graphviz")
        sys.exit(1)

    repo_path = sys.argv[1]
    output_format = sys.argv[2].lower() if len(sys.argv) > 2 else "graphviz"

    print(f"Analyzing repository: {repo_path}")
    results = analyze_repo(repo_path)

    print("=" * 80)
    print(f"REPO ANALYSIS: {repo_path}")
    print("=" * 80)

    print(f"\nTotal Python files: {results['summary']['total_files']}")
    print(f"Total unique imports: {len(results['summary']['total_imports'])}")
    print(f"Internal imports: {len(results['summary']['internal'])}")
    print(f"Third-party imports: {len(results['summary']['third_party'])}")

    print("\n" + "=" * 80)
    print("TOP 20 MOST IMPORTED MODULES")
    print("=" * 80)

    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            import_counts[imp] += 1

    for imp, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        prefix = "🟢" if imp in results["summary"]["internal"] else "🔴"
        print(f"{prefix} {imp}: {count}")

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    if output_format == "graphviz" or output_format == "svg":
        create_graphviz_visualization(results, KretConstants.DATA_DIR / "import_dependencies.svg")
    elif output_format == "matplotlib" or output_format == "png":
        create_matplotlib_visualization(results, KretConstants.DATA_DIR / "import_dependencies.png")
    elif output_format == "interactive" or output_format == "html":
        if not nx:
            print("Installing networkx...")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
            import networkx as nx
        create_html_interactive(results, KretConstants.DATA_DIR / "import_dependencies.html")
    else:
        print(f"Unknown format: {output_format}")
        print("Try: graphviz, matplotlib, or interactive")
