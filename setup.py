from setuptools import setup

with open("requirements.txt", "r") as f:
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
	name="reinforcelab",
	version='0.0.0',
	escription="Standardized and documented solutions for Reinforcement Learning tasks",
    url="https://github.com/factoredai/ReinforceLab",
    author="Factored",
    packages=["reinforcelab"],
    install_requires=requires,
    python_requires=">=3.9",
)