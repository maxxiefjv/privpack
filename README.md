# As Contributor

The recommended steps to develop on this project include:

1. Creating a virtual environment which runs python3.
2. Install the package locally with the development packages.

These steps are represented by the following commands:

```bash 
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

NOTE: python3 is currently required for this project.

# Testing

For testing purposes one should use the pytest library included in the development requirements of this package. Create your test in the corresponding test file and run the `$ pytest <path/to/test/file>` command in the root directory of this project.

# Example program
Currently the only way to run the code is by using the pytest module. There is an example program under development.