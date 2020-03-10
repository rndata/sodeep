from setuptools import setup, find_packages


setup(
    name="sodeep",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
        ],
    },
    python_requires=">=3.7,<3.8",
    install_requires=[
        "toolz ~= 0.9",
        "numpy ~= 1.18.1",
        "tensorflow ~= 2.1.0",
        "beautifulsoup4 ~= 4.8.2",
        "regex ~= 2020.2.20",
    ],
    extras_require={
        "gpu": [
            "tensorflow-gpu ~= 2.1.0"
        ],
        "run": [
            "fire ~= 0.2",
        ]
    }
)
