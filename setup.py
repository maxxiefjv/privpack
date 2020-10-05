import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="privpack-maxxiefjv",
    version="0.0.2",
    author="Max Vasterd",
    author_email="vasterd97@gmail.com",
    description="A small package providing the tools to learn a GAN to preserve privacy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxxiefjv/Privpack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'numpy',
        'scipy'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'sphinx'
        ]
    }
)
