# Virtual Environment
before installing any package, first create virtual environment.
You can create a Python virtual environment using the venv module, which comes with Python 3.
Here are the steps to create a virtual environment:

```
1. Open a terminal or command prompt.
2. Navigate to the directory where you want to create the virtual environment.
3. Run the following command to create a new virtual environment:
        $ python3 -m venv myvenv
    
This will create a new directory called myenv in your current directory, which will contain the virtual environment.
4. Activate the virtual environment by running the following command:
        $ source myvenv/bin/activate
Note that the command above is for Unix-based systems. If you're using Windows, the command to activate the virtual environment is:
        myenv\Scripts\activate.bat
5. Once the virtual environment is activated, you can install packages using pip, and they will be installed only in the virtual environment, not globally.
To exit the virtual environment, you can run the following command:
        $ deactivate

```

## required packages
```
$ pip install pandas
$ pip install -U scikit-learn
$ pip install matplotlib
$ pip install seaborn
$ pip install imbalanced-learn
$ pip install pytest
```