from setuptools import setup

setup(
    name="seismoai-project",
    version="0.1",
    py_modules=[
        "seismoai_io",
        "seismoai_model",
        "seismoai_xai"
    ],
    install_requires=[
        "numpy",
        "segyio",
        "scikit-learn",
        "shap",
        "matplotlib"
    ]
)
