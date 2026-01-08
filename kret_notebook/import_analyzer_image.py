"""
Analyze import dependencies in a Python repo and generate SVG visualization.
Usage: python import_graph_viz.py /path/to/repo
"""

import os
import sys

from dotenv import load_dotenv

# Load .env file from kretsinger directory
load_dotenv("/Users/Akseldkw/coding/kretsinger/.env")

# Add to Python path
pythonpath = os.getenv("PYTHONPATH")
if pythonpath:
    for path in pythonpath.split(":"):
        sys.path.insert(0, path)

import ast
from collections import defaultdict
from pathlib import Path

from kret_utils.constants_kret import KretConstants

try:
    import matplotlib.pyplot as plt
    import networkx as nx

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
        with open(filepath, encoding="utf-8") as f:
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


def analyze_repo(repo_path: str | Path) -> dict:
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


def create_graphviz_visualization(results: dict, output_file: str | Path = "import_dependencies.svg"):
    """Create SVG using Graphviz - local packages only."""
    if graphviz is None:
        print("ERROR: graphviz not installed. Install with: pip install graphviz")
        return False

    dot = graphviz.Digraph(comment="Import Dependencies (Local Only)", format="svg")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

    # Only use internal imports
    internal = results["summary"]["internal"]

    # Count only internal imports
    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            if imp in internal:  # Only count local packages
                import_counts[imp] += 1

    # Get top 50 most imported local modules
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    top_import_names = {imp for imp, _ in top_imports}

    # Add nodes - all are internal/local
    for imp, count in top_imports:
        dot.node(imp, f"{imp}\n({count})", fillcolor="lightgreen")

    # Add edges - only for local packages
    edge_count = 0
    for module, imports in results["imports"].items():
        for imp in imports:
            if imp in top_import_names and edge_count < 100:  # Limit edges
                dot.edge(module, imp, weight=str(import_counts[imp]))
                edge_count += 1

    # Save - ensure we're in the output directory so /lib gets created there
    output_path = Path(output_file) if not isinstance(output_file, Path) else output_file
    output_dir = output_path.parent
    output_name = str(output_path.stem)

    # Change to output directory before rendering
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        dot.render(output_name, cleanup=True)
    finally:
        os.chdir(original_cwd)

    print(f"✓ SVG saved to: {output_file} (local packages only)")
    return True


def create_matplotlib_visualization(results: dict, output_file: str | Path = "import_dependencies.png"):
    """Create PNG using matplotlib and networkx - local packages only."""
    if not nx or not plt:
        print("ERROR: networkx/matplotlib not installed. Install with: pip install networkx matplotlib")
        return False

    # Build graph
    G = nx.DiGraph()

    # Only use internal imports
    internal = results["summary"]["internal"]

    # Count only internal imports
    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            if imp in internal:  # Only count local packages
                import_counts[imp] += 1

    # Get top 30 local imports
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    top_import_names = {imp for imp, _ in top_imports}

    # Add nodes - all are local
    for imp, count in top_imports:
        G.add_node(imp, count=count)

    # Add edges - only for local packages
    edge_count = 0
    for module, imports in results["imports"].items():
        for imp in imports:
            if imp in top_import_names and edge_count < 80:
                G.add_edge(module, imp)
                edge_count += 1

    # Draw
    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # All nodes are internal (green)
    node_colors = ["lightgreen" for _ in G.nodes()]

    # Node sizes based on import count
    node_sizes = [import_counts.get(node, 1) * 50 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, width=0.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    plt.title("Python Import Dependency Graph\n(Local Packages Only)", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ PNG saved to: {output_file} (local packages only)")
    return True


def create_html_interactive(results: dict, output_file: str | Path = "import_dependencies.html"):
    """Create interactive HTML visualization - local packages only."""
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

    # Only use internal imports
    internal = results["summary"]["internal"]

    # Count only internal imports
    import_counts = defaultdict(int)
    for imports in results["imports"].values():
        for imp in imports:
            if imp in internal:  # Only count local packages
                import_counts[imp] += 1

    # Get top local imports
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:40]
    top_import_names = {imp for imp, _ in top_imports}

    # Add nodes - all are local (green)
    for imp, count in top_imports:
        G.add_node(imp, title=f"{imp} (imported {count} times)", color="#90EE90", size=min(count * 3, 50))

    # Add edges - reversed direction (imp -> module) to show what depends on what
    for module, imports in results["imports"].items():
        for imp in imports:
            if imp in top_import_names:
                G.add_edge(imp, module)  # Reversed: now shows imp -> module

    # Create pyvis network with more height for the graph
    net_graph = net.Network(height="80vh", width="100%", directed=True)
    net_graph.from_nx(G)

    # Customize physics using the correct API
    net_graph.toggle_physics(True)
    net_graph.show_buttons(filter_=["physics"])

    # Save - ensure we're in the output directory so /lib gets created there
    output_path = Path(output_file) if not isinstance(output_file, Path) else output_file
    output_dir = output_path.parent

    # Change to output directory before saving (pyvis creates /lib folder in cwd)
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        net_graph.show(str(output_path.name), notebook=False)

        # Post-process HTML to adjust layout and make it resizable
        html_path = output_dir / output_path.name
        with open(html_path) as f:
            html_content = f.read()

        # Add custom CSS for better proportions and resizable divider
        custom_css = """
             body { margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
             #mynetwork { flex: 1; min-height: 60vh; border: 1px solid lightgray; position: relative; }
             #config { height: auto; max-height: 35vh; overflow-y: auto; border-top: 3px solid #ccc;
                       resize: vertical; padding: 10px; background: #f5f5f5; }
        """
        # Find the last </style> tag and insert custom CSS before it
        last_style_end = html_content.rfind("</style>")
        if last_style_end != -1:
            html_content = html_content[:last_style_end] + custom_css + html_content[last_style_end:]

        with open(html_path, "w") as f:
            f.write(html_content)

    finally:
        os.chdir(original_cwd)

    print(f"✓ Interactive HTML saved to: {output_file} (local packages only)")
    print("  - Arrows reversed: now show what each module depends on")
    print("  - Config panel is resizable (drag the top border)")
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
