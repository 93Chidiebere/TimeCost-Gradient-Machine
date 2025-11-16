from setuptools import setup, find_packages

setup(
    name="tcgm",
    version="0.1.0",
    description="TimeCost Gradient Machine — a time-aware, cost-sensitive boosting algorithm for finance.",
    author="Chidiebere V. Christopher",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.5",
        "scikit-learn>=1.1",
        "joblib>=1.1",
        "matplotlib>=3.5"
    ],
    python_requires=">=3.8",
)