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
$ pip install pandas - for data processing and manipulation
$ pip install -U scikit-learn - for machine learning algorithms and model evaluation 
$ pip install matplotlib - for data visualization
$ pip install seaborn - for statistical graphics
$ pip install imbalanced-learn - for handling imbalanced datasets
$ pip install pytest - for writing and running tests for code
$ pip install numpy - for numerical computations
$ pip install tkinter - for creating graphical user interface (GUI) applications in Python

```

# Stroke Prediction System

This is a machine learning model that uses logistic regression to model the relationship between the predictor variables and the stroke outcome. 
The model uses Python and its libraries such as Pandas, Scikit-learn, and Matplotlib for data processing, modeling, and evaluation. It predicts the likelihood of an individual having a stroke based on demographic, lifestyle, and health-related factors. The main risk factors used were age, BMI and average glucose level. Other factors included in the model are heart-disease history, hypertension, place of residence(urban/rural), smoking factors, occupation and gender.

```

## User Stories
As a medical professional, you can  use this model to:

Identify patients who are at high risk of stroke and recommend preventative measures based on the mosel.
Improve the accuracy of stroke risk prediction and reduce false positives/negatives.
Better understand the key risk factors for stroke and how they contribute to stroke risk.

```

## Getting Started

To use this model, you will need to have Python and the required libraries installed on your computer. You can install the required libraries by running the following command in your terminal:

```
pip install -r requirements.txt
```

Once the libraries are installed, you can run the `stroke_prediction_system.py` file to train the model and generate predictions. The script will prompt you to provide input values for the predictor variables, such as age, gender, smoking status, and hypertension. Upon running the file, a visualization of risk factors for high stroke probability pops up on a graph. Once you exit the graph, the Stroke Prediction System GUI pops up and is ready for prediction.

## Dataset

The dataset used to train this model is the Stroke Prediction Dataset from Kaggle, which contains over 4,000+ records of patient information, including age, gender, smoking status, hypertension, heart disease, and BMI. (All references/sources are cited at the end of the README.md file and powerpoint presentation)

## Model Performance

 This model is reliable and accurate in predicting stroke risk based on the given input variables. (Age, BMI, avg glucose level,etc)

## Challenges

## Logistic Regresssion Model

## License
This project is not yet licensed. It is an educational model. All sources are properly cited and credit is given in the References tab.

## Acknowledgments

- Kaggle for providing the Stroke Prediction Dataset
- Python and its libraries for making machine learning accessible and easy to use
- The research community, healthcare professionals for their endless contributions to stroke prevention and treatment.
- My instructors Bryan Beach, Trevor Unaegbu & Melissa Robinson for their hardwork and dedication to instilling knowledge.
- Talent Advocate Managers Mark Bachmann & Priscilla Jones for
- My recruiter Magdaleny Soberanis
- Entire TEKsystems-Takeda Team for their resources and time during the entire duration of the training program 
- My Classmates for allowing me to learn from them, persevering and growing together

## References for Project
- Stroke Facts | cdc.gov. (2023, May 4). Centers for Disease Control and Prevention. https://www.cdc.gov/stroke/facts.htm

- Mechatronixs. (2023). Prediction with 7-Classification Models | ROC AUC. Kaggle. https://www.kaggle.com/code/mechatronixs/prediction-with-7-classification-models-roc-auc

- TEDx Talks. (2020, March 6). Can We More Accurately Predict Heart Attacks and Strokes? | Dr. David Farrell | TEDxMcMinnville [Video]. YouTube.      https://www.youtube.com/watch?v=bfsy0l0APYA

- Feigin, V. L., Brainin, M., Norrving, B., Martins, S. C. O., Sacco, R. L., Hacke, W., Fisher, M., Pandian, J. D., & Lindsay, P. (2022). World Stroke Organization (WSO):   Global Stroke Fact Sheet 2022. International Journal of Stroke, 17(1), 18–29. https://doi.org/10.1177/17474930211065917

- What is Logistic regression? | IBM. (n.d.). https://www.ibm.com/topics/logistic-regression

- PEP 8 – Style Guide for Python Code | peps.python.org. (n.d.). https://peps.python.org/pep-0008/

