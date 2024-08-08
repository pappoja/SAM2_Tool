from setuptools import setup, find_packages

setup(
    name='sam2_annotation_tool',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'matplotlib',
        'pillow',
        'ipywidgets',
        'ipympl',
        'torch',
    ],
)
