from setuptools import setup, find_packages

setup(
    name="omega-mesh-engine",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.13.0",
        "flask>=2.2.0",
        "python-dateutil>=2.8.0",
        "pdfplumber>=0.7.0",
    ],
    extras_require={
        'cuda': [
            "cupy-cuda12x>=12.0.0",
            "numba>=0.57.0",
            "torch>=2.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Casey Jussaume",
    description="Omega Mesh Financial Engine with CUDA Acceleration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
    ],
) 