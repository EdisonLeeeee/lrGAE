from setuptools import find_packages, setup

VERSION = '0.1.0'
url = 'https://github.com/EdisonLeeeee/lrGAE'

install_requires = [
    'tqdm',
    'scipy',
    'numpy',
    'scikit_learn',
    'torch_geometric',
]

full_requires = [
    'numba',
    'pandas',
    'matplotlib',
    'networkx>=2.3',
    'texttable',
    'pandas',
    # 'gensim>=3.8.0',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

dev_requires = test_requires + [
    'pre-commit',
]

setup(
    name='lrgae',
    version=VERSION,
    description='Graph autoencoder benchmark',
    author='Jintang Li',
    author_email='lijt55@mail2.sysu.edu.cn',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, VERSION),
    keywords=[
        'torch_geometric',
        'pytorch',
        'benchmark',
        'geometric-auto-encoders',
        'graph-neural-networks',
    ],
    python_requires='>=3.7',
    license="MIT LICENSE",
    install_requires=install_requires,
    extras_require={
        'full': full_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(exclude=("examples", "imgs", "benchmark", "test")),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
