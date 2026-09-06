from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README_PATH = ROOT / "README.md"
REQUIREMENTS_PATH = ROOT / "requirements.txt"

long_description = (
    README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
)
install_requires = [
    line.strip()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="kathdb",
    version="0.1.0",
    description="KathDB: A transparent multimodal DBMS powered by ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Guorui Xiao",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={"kathdb": ["**/*.md", "**/*.txt", "generated_fn/.gitkeep"]},
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={"console_scripts": ["kathdb=kathdb.cli:main"]},
)
