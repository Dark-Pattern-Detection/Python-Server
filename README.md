1. To clone the repository:  `git clone https://github.com/Dark-Pattern/Python-Server.git`
2. Go into the cloned folder: `cd  Python-Server`
3. Run in the terminal: `pip install virtualenv`
4. Run command in `Python-Server` folder to create a `env` folder: `python -m virtualenv env`
5. Activate the virtual environment, after running this `(env)` will appear in the start of command line.: `.\env\Scripts\activate.ps1`
6. To install all the requirements in the virtual environment: `pip install -r requirements.txt`
7. Download the model from google drive and extract it into `Python-Server` folder model folder name as `model.yesno.update`
8. It will run the Python server on `http://localhost:3000`: `python app.py`
