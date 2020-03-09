from setuptools import setup, find_packages


setup(
    name="sodeep",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
        ],
    },
    python_requires=">=3.7,<3.9",
    install_requires=[
        "toolz ~= 0.9",
        "fire ~= 0.2",
        "numpy ~= 1.18.1",
        "tensorflow ~= 2.1.0",
        "beautifulsoup4 ~= 4.8.2",
        "regexp ~= 0.1",
    ],
    extra_require={
        "gpu": [
            "tensorflow-gpu ~= 2.1.0"
        ]
    }
)
