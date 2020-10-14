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
An example program is available called privpack-app. To run this program open your python environment and run `$ python privpack-app -h` in the root folder of this project. This will show all the options available for running the example app. At this moment
two Privacy-Preserving networks have been implemented in the example program: Binary and Gaussian. 