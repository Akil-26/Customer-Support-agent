# Minimal setup so `import agent` works from project root
# Run once: pip install -e .
from setuptools import setup, find_packages

setup(
    name="amazon-support-agent",
    version="0.1.0",
    packages=find_packages(),
)
