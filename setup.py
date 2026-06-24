from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements_path = Path(__file__).with_name("requirements.txt")
    requirements = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line.split("#", 1)[0].strip())
    return requirements

setup(
    name="news_project",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "news-recommend=src.run_recommend_cli:main",
        ],
    },
)
