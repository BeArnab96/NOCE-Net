from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent

setup(
    name="noce-net",
    version="0.1.0",
    description="NOCE-Net: nested operational context encoding for battery voltage-response forecasting",
    python_requires=">=3.9",
    packages=find_packages(exclude=["scripts", "scripts.*"]),
    install_requires=(HERE / "requirements.txt").read_text().splitlines(),
)
