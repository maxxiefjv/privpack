# Running the project

Currently to run this project one has to create a virtual python environment. If the venv pip package
is not yet installed use: `pip install virtualenv`. After installing virtualenv, create a virtual environment, install the required packages, and run it the code. These steps are represented by the following commands:

```bash 
virtualenv venv
source venv/bin/activate
pip3 install -r ./requirements.txt
python3 bin --help
```
The last command will show all the possible commands one can run.

NOTE: python3 is currently required for this project. To force a venv which uses python3, run: `virtualenv -p {PATH_TO_PYTHON3} venv`
instead.

The `python3 bin` command will run the "\_\_main__.py" in the {PROJECT_ROOT}/bin folder.