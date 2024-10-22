from setuptools import setup

setup(
    name="Physics Simulations",
    version="0.1",
    description="Repository for web-based physics simulations",
    package_dir={"": "src"},
    packages=["web", "schrodinger", "pendulum"]
)