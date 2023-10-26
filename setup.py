from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf8')

setup(
    name='autoencoders',
    version='0.0.1',
    keywords=['encoder', 'autoencoder', 'deep learning', 'pytorch'],
    description='autoencoders for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT Licence',
    url='https://github.com/Jyonn/autoencoders',
    author='Jyonn Liu',
    author_email='i@6-79.cn',
    platforms='any',
    packages=find_packages(),
    install_requires=[
    ],
)
