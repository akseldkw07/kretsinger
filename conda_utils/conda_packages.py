import json
import os
import subprocess
import sys
import typing as t
from subprocess import CompletedProcess

from packaging.version import parse  # Used for robust version comparison


class CondaUtils:
    JSON_NAME = "min_conda_versions.json"
    # region Conda Packages
    CORE: list[str] = [
        "numpy",
        "pandas",
        "pyarrow",
        "python-dotenv",
        "jupyter",
        "notebook",
        "jupyterlab",
        "polars",
        "pre-commit",
        "black",
        "flake8",
        "conda-tree",
        "numba",
        "ipykernel",
        "jax",  # high-performance numerical computing. Like numpy but faster. Vectorized
        "pytest",
        "pathvalidate",
        "isort",
    ]
    FINANCE = [
        "yfinance",
        "pandas-datareader",
        "pmdarima",  # timeseries
    ]
    VISUALIZATION = [
        "plotly",
        "bokeh",
        "matplotlib",
        "dash",
        "panel",
        "dash-bootstrap-components",
        "dash-core-components",
        "dash-html-components",
        "pydot",
        "pydotplus",
        "ipywidgets",
        "pygraphviz",  # draw graphs
        "graphviz",
        "drawdata",  # https://python.plainenglish.io/drawdata-the-python-library-you-didnt-know-you-needed-b8c2f2ff328b
        "pydeps",
        # "streamlit",  # web apps, dashboards, and data storytelling # TODO check 1.56.0 released
    ]
    STATS = [
        # "abess",  # Algorithm for Best Element-wise Subset selection. don't add, pin to numpy 1.26
        "scikit-learn",
        "seaborn",
        "statsmodels",
        "scipy",
        # "arviz_plots",
        # "arviz_base",
        # "arviz_stats",
    ]
    BAYESIAN_STATS = [
        "pymc",  # bayesian modeling, MCMC, VAI
        "arviz",  # Bayesian analysis and visualization
        # "scikit-surprise",  # recommender systems. https://surpriselib.com/] ----- don't add, pin to numpy 1.26
        # 'lightfm',  # hybrid recommendation algorithms ----- don't add, pin to numpy 1.26 AND python 3.11
        "implicit",  # collaborative filtering for implicit datasets
        "BoTorch",  # Bayesian optimization built on PyTorch
    ]
    R_STATS = ["sympy", "siuba"]
    NLP = [
        "nltk",
        #   "gensim", #conflicts with numpy > 2.0.0
        "spacy",
        "textblob",
        "transformers",
        "sentence-transformers",
        "evaluate",
    ]
    REDDIT = ["praw", "vaderSentiment"]
    ML = [
        "xgboost",
        "lightgbm",
        "catboost",
        # "fbprophet",  # timeseries forecasting, facebook NOTE doesn't exist
        "wandb",  # weights and biases
        "optuna-dashboard",
        "optuna",
        "optuna-integration",
    ]
    NN_DL = [
        "tensorflow",
        "keras",  # tensorflow wrapper
        # "tf-keras", # NOTE not compatible with latest
        "pytorch",
        "lightning",
        # 'MLFlow',
        "fastai",  # pytorch wrapper, more configurable than keras
        "kaggle",
        "kagglehub",
        "autograd",
        "torchview",
        "netron",  # visualize neural network architectures
        "onnxscript",
        # 'thop', # for calculating FLOPs and params in pytorch models - PIP ONLY
        "transformers",
        # "vllm", uv pip install vllm
        "accelerate",
        "peft",  # parameter-efficient fine-tuning for LLMs, like LoRA. NOTE: this is a bit finicky to install on macos, but pip works fine. Conda package is available but not always up to date, so we'll install via pip in the setup script.
        # "bitsandbytes",  # for 8-bit optimizers and quantization in PyTorch, often used with LLMs (no MacOS)
    ]
    CV = [
        "opencv",
        "pillow",
        "scikit-image",
        "torchvision",
        "timm",
        # "faster-coco-eval",  # this is tough on macos for some reason, but pip works
        "pycocotools",
    ]
    COLUMBIA = [
        "opencv-python",
        "pycocotools",
        "timm",
    ]
    LLM_QUERY = [
        # REQUIRE GPU
        "transformers",  # for LLMs
        "sentencepiece",
        "accelerate",
        "optimum",
    ]
    LINALG_OPTIMIZATION = [
        "pulp",  # optimization
        "sympy",
        "cvxpy",  # optimization
    ]
    GRAPHS = ["graphviz", "networkx", "osmnx", "pytorch_geometric"]
    RL = ["gymnasium", "pygame"]
    CI = ["tigramite"]  # causal inference, time series
    LLMS = [
        "anthropic",
        "openai",
    ]  # for querying LLMs. NOTE: these are not the same as the transformers library, which is for running LLMs locally. These are for querying APIs like OpenAI and Anthropic.
    MISC = ["openpyxl", "youtube-transcript-api"]
    PIP_ONLY = [
        "faster-coco-eval",
        "thop",
    ]  # packages that are only available via pip and not conda, will be installed separately in the setup script

    ALL = (
        CORE
        # + FINANCE
        + VISUALIZATION
        + STATS
        + BAYESIAN_STATS
        # + R_STATS
        + NLP
        # + REDDIT
        + ML
        + NN_DL
        + CV
        # + COLUMBIA
        # + LLM_QUERY
        + LINALG_OPTIMIZATION
        + GRAPHS
        + RL
        + CI
        + LLMS
        + MISC
        # + PIP_ONLY
    )

    PINS_OVERRIDE = {}
    PINNED_PACKAGES = ["numpy", "pandas", "polars", "scikit-learn", "datasets"]

    # endregion
    @staticmethod
    def conda_packages_to_str(
        packages: t.Iterable[str] = ALL,
        pin_loc: str | t.Literal[False] = JSON_NAME,
        pins_override: dict[str, str] = PINS_OVERRIDE,
        version: str = "3.12",
    ):
        packages = set(packages)
        if pin_loc is not False:
            with open(pin_loc) as f:
                loaded: dict[str, str] = json.load(f)
            pins = loaded | pins_override
            print(f"Loaded pins: {pins}")
        else:
            pins = pins_override
            print(f"Using override pins: {pins}")

        packages = sorted(packages, key=lambda x: (x not in pins, x))
        packages_w_pins = [package if package not in pins else f"'{package}>={pins[package]}'" for package in packages]
        packages_str = " ".join(packages_w_pins)

        prepend = f"mm create -n kret_{version.replace('.', '')} python={version} "
        suffix = " --yes"

        print(prepend + packages_str + suffix)

    @staticmethod
    def get_latest_conda_versions(packages: list[str] = PINNED_PACKAGES) -> dict[str, str | None]:
        """
        Finds the latest available version for a list of Conda packages by
        querying the Conda CLI.

        Args:
            packages (list): A list of package names (e.g., ['numpy', 'pandas']).

        Returns:
            dict: A dictionary where keys are package names and values are their
                latest version strings. If a package is not found or an error
                occurs for a specific package, its value will be None.
        """
        latest_versions = {}
        command = []
        result = CompletedProcess([], 0)

        for package_name in packages:
            try:
                # *** CHANGE HERE: Call 'conda' directly, not 'python -m conda' ***
                command = ["conda", "search", "--json", package_name]

                result = subprocess.run(command, capture_output=True, text=True, check=True)

                data = json.loads(result.stdout)

                if package_name in data and data[package_name]:
                    versions = []
                    for entry in data[package_name]:
                        if "version" in entry:
                            try:
                                versions.append(parse(entry["version"]))
                            except Exception as e:
                                print(
                                    f"Warning: Could not parse version '{entry.get('version')}' for {package_name}: {e}",
                                    file=sys.stderr,
                                )
                    if versions:
                        latest_version = max(versions)
                        latest_versions[package_name] = str(latest_version)
                    else:
                        latest_versions[package_name] = None
                        print(f"No parsable versions found for package: {package_name}", file=sys.stderr)
                else:
                    print(f"Package '{package_name}' not found in Conda search results.", file=sys.stderr)
                    latest_versions[package_name] = None

            except subprocess.CalledProcessError as e:
                print(f"Error searching for {package_name} (Conda command failed):", file=sys.stderr)
                print(f"  Command: {' '.join(command)}", file=sys.stderr)
                print(f"  Return Code: {e.returncode}", file=sys.stderr)
                print(f"  Stdout: {e.stdout.strip()}", file=sys.stderr)
                print(f"  Stderr: {e.stderr.strip()}", file=sys.stderr)
                latest_versions[package_name] = None
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON output for {package_name}: {e}", file=sys.stderr)
                print(f"  Raw output (first 500 chars): {result.stdout[:500].strip()}", file=sys.stderr)
                latest_versions[package_name] = None
            except FileNotFoundError:
                print(
                    "Error: 'conda' command not found. Make sure Conda is installed and its base environment is activated or in your system's PATH.",
                    file=sys.stderr,
                )
                return {pkg: None for pkg in packages}
            except Exception as e:
                print(f"An unexpected error occurred for {package_name}: {e}", file=sys.stderr)
                latest_versions[package_name] = None

        return latest_versions

    @staticmethod
    def update_latest_versions_json(
        packages: list[str] = PINNED_PACKAGES,
        new_versions: dict[str, str | None] | None = None,
        file_path: str = JSON_NAME,
    ):
        """
        Fetches the latest Conda package versions and updates a JSON file.
        Also prints a comparison before saving.

        Args:
            packages (list[str]): List of package names to check.
            file_path (str): The path to the JSON file to read/write.
        """
        # 1. Read existing versions (if file exists)
        existing_versions = {}
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    existing_versions = json.load(f)
                print(f"Successfully loaded existing versions from '{file_path}'.")
            except json.JSONDecodeError:
                print(f"Warning: Existing '{file_path}' is not valid JSON. Starting fresh.", file=sys.stderr)
                existing_versions = {}
            except Exception as e:
                print(f"Error reading existing '{file_path}': {e}. Starting fresh.", file=sys.stderr)
                existing_versions = {}
        else:
            print(f"'{file_path}' not found. A new file will be created.")

        # 2. Get new latest versions
        print("\nFetching current latest versions from Conda CLI...")
        new_latest_versions = new_versions or CondaUtils.get_latest_conda_versions(packages)

        # 3. Compare existing with new
        print("\n--- Version Comparison ---")
        found_differences = False

        # Check for changes in existing packages or new packages
        for pkg, new_ver in new_latest_versions.items():
            old_ver = existing_versions.get(pkg)

            if old_ver is None and new_ver is None:
                # This could happen if a package was requested but not found previously and still not found
                continue  # Skip reporting if both old and new are None (e.g. package doesn't exist)
            elif old_ver is None and new_ver is not None:
                print(f"  New Package: {pkg} -> {new_ver}")
                found_differences = True
            elif old_ver is not None and new_ver is None:
                print(f"  Removed/Error: {pkg} (was {old_ver}) -> Now not found/error")
                found_differences = True
            elif old_ver != new_ver:
                print(f"  Updated: {pkg}: {old_ver} -> {new_ver}")
                found_differences = True
            # else: old_ver == new_ver, no change, do nothing

        # Check for packages that existed but are no longer requested
        for pkg, old_ver in existing_versions.items():
            if pkg not in new_latest_versions:
                print(f"  No Longer Tracked: {pkg} (was {old_ver})")
                found_differences = True

        if not found_differences:
            print("  No changes detected for the tracked packages.")
        print("------------------------")

        # 4. Save the new latest versions to the JSON file
        print(f"\nSaving new latest versions to '{file_path}'...")
        with open(file_path, "w") as f:
            json.dump(new_latest_versions, f, indent=4)
        print("Save complete.")
