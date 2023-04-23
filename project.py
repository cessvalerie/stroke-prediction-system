import datetime
import getpass
import os
import sys
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from matplotlib.gridspec import GridSpec #data viz
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE

################### import for gui #########################
import tkinter as tk
from tkinter import *
############################################################

############ constant #################################################
system='windows'
dataset=''
time_string_format='%d-%m-%Y %H:%M:%S'
filename = 'logistic_regression_model.sav'

scaler1 = StandardScaler()
scaler2 = StandardScaler()


########## gui related constant ########
number_of_field =0
start_position = 60
position_diff = 40

HYPERTENSION_OPTIONS=["YES", "NO"]
WORK_OPTIONS=["Private","Self-employed","Govt_job","children","Never_worked"]
RESIDENCE_OPTIONS=["Urban", "Rural"]
SMOKING_OPTIONS=["Formaly smoked","never smoked","smokes","Unknown"]
GENDER_OPTIONS = ["Male","Famle","Other"]
dropdown_list=[]
entry_field_list=[]
########################################################################


######################################### methods for gui ############################################3
def submitForm():

    data = {}
    x = entry_field_list[0].get()
    if(len(x)>0):
        data["age"] = float(x)

   
    x = dropdown_list[1].get()
    if(x=="YES"):
        data["hypertension"]=1
    else:
        data["hypertension"]=0



    x = dropdown_list[2].get()
    if(x=="YES"):
        data["heart_disease"]=1
    else:
        data["heart_disease"]=0



    x = dropdown_list[3].get()
    if(x=="YES"):
        data["ever_married"] = 1
    else:
        data["ever_married"] = 0

    # Residence Type
    x = dropdown_list[5].get()
    if(x== "Urban"):
        data["Residence_type"] = 1
    else:
        data["Residence_type"] = 0



    x = entry_field_list[1].get()
    if(len(x)>0):
        data["avg_glucose_level"] = float(x)

    x = entry_field_list[2].get()
    if(len(x)>0):
        data["bmi"] = float(x)


    x = dropdown_list[0].get()
    data["gender"] = x

    x = dropdown_list[4].get()
    data["work_type"] = x

    # Smoking
    x = dropdown_list[6].get()
    data["smoking_status"] = x

    list =[data]
    print(list)

    result = predict(list)
    
    print(result)
    label_result.config(text =result)
    if(result=='Positive'):
        label_result.config(fg='green')
    else:
        label_result.config(fg='red')
    


def dropdown(tk, canvas,label_title, OPTIONS):
    #label
    global number_of_field, start_position, position_diff
    number_of_field+=1

    label = tk.Label(root, text=label_title)
    label.config(font=('helvetica', 10), width=50, anchor="w",justify=LEFT)
    canvas.create_window(250, start_position+(number_of_field*position_diff), window=label)

    # input field
    variable = StringVar(root)
    variable.set(OPTIONS[0]) # default value
    dd = OptionMenu(root, variable, *OPTIONS)
    canvas.create_window(400, start_position+(number_of_field*position_diff), window=dd)
    dropdown_list.append(variable)



def entry_field(tk, canvas, label_title):
    global number_of_field, start_position, position_diff

    number_of_field+=1

    label = tk.Label(root, text=label_title)
    label.config(font=('helvetica', 10), width=50, anchor="w",justify=LEFT)
    canvas.create_window(250, start_position+(number_of_field*position_diff), window=label)

    entry = tk.Entry(root) 
    canvas.create_window(400, start_position+(number_of_field*position_diff), window=entry)
    entry_field_list.append(entry)


def gui():
    global root, number_of_field, label_result, root
    root= tk.Tk()
    root.title("Stroke prediction system")

    canvas1 = tk.Canvas(root, width=700, height=600, relief='raised')
    canvas1.pack()

    label1 = tk.Label(root, text='Please give information to get the result:')
    label1.config(font=('helvetica', 12))
    canvas1.create_window(200, 25, window=label1)
    
    
    # input form fields
    label_title = "Enter patient age:"
    entry_field(tk,canvas1, label_title)
    
    # gender dropdown  
    label_title = "Enter Patient Gender:"
    dropdown(tk,canvas1,label_title, GENDER_OPTIONS)
    
    
    # hyper tension dropdown  
    label_title = "Do patient have Hypertension:"
    dropdown(tk, canvas1, label_title, HYPERTENSION_OPTIONS)
    
    label_title="Heart disease history:"
    dropdown(tk,canvas1,label_title, HYPERTENSION_OPTIONS)
    
    
    label_title="marrial status (married/unmarried):"
    dropdown(tk,canvas1,label_title, HYPERTENSION_OPTIONS)
    
    label_title="Enter work type:"
    dropdown(tk, canvas1, label_title,WORK_OPTIONS)
    
    
    label_title = "Enter avg glucose level:"
    entry_field(tk,canvas1, label_title)
    
    
    label_title = "Enter BMI (Body Mass Index):"
    entry_field(tk,canvas1, label_title)
    
    
    label_title="Enter residence type (Urban/Rural):"
    dropdown(tk, canvas1, label_title,RESIDENCE_OPTIONS)
    
    
    label_title = "Smoking status"
    dropdown(tk, canvas1, label_title,SMOKING_OPTIONS)
    

    number_of_field+=1
    button1 = tk.Button(root,text='Predict', command=submitForm, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas1.create_window(200, start_position+(number_of_field*position_diff), window=button1)

    number_of_field+=1
    label_result = tk.Label(root, text="", font=('helvetica', 14, 'bold'))
    canvas1.create_window(300, start_position+((number_of_field)*position_diff), window=label_result)
    root.mainloop()

################################################################################################################





def detect_system():
    global system
    # sys.platform.startswith('darwin') is required for mac-os. this os is out of support till now.
    if sys.platform.startswith('win'):
        system ='windows'
    elif sys.platform.startswith('linux'):
        system='linux'
    else:
        system='unknown'


def read_dataset():
    global dataset
    for dirname, _, filenames in os.walk('dataset'):
        for filename in filenames:
            dataset = os.path.join(dirname, filename)
            print("Dataset location: {}".format(dataset))
    
    data = pd.read_csv(dataset)
    df = data.copy()
    #print(df.head())
    return df


# Checking the Data
def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head(head))
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)



# identify Column types
def identify_cols_types(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]


    print("\n" + 20*"-" + "Dataset columns types".center(20) + 20*"-")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car



# Start Outliers Threshold
def outlier_th(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False  

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit    


def handle_outliers(dataframe, num_cols):

    print("\n" + 20*"-" + "Handle Outlier".center(20) + 20*"-")
    print("\n" + "# Check Outlier".center(10) + 10*"-")

    # check for outlier
    for col in num_cols:
        print(col, check_outlier(dataframe, col))

    
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    print("\n" + "# Check Outlier after replacing with thresholds".center(10) + 10*"-")
    # check for outlier again after replacing
    for col in num_cols:
        print(col, check_outlier(dataframe, col))

# End Outliers Threshold



# Start Handling Missing Values
def handle_missing_values(dataframe):
    print("\n" + 20*"-" + "Handle Missing values".center(20) + 20*"-")

    #na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    na_columns = []
    for col in dataframe.columns:
        if(dataframe[col].isnull().sum()>0):
            na_columns.append(col)


    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio over dataset'])
    print(missing_df, end="\n")

    # fill missing value by mean value
    for col in na_columns:
        mean = dataframe[col].mean()
        dataframe[col].fillna(mean, inplace=True)

    
    # checking again for null value
    na_columns = []
    for col in dataframe.columns:
        if(dataframe[col].isnull().sum()>0):
            na_columns.append(col)
    print("number of null columns after handling: {}".format(len(na_columns)))
    return
# End Handling Missing Values


# data visualization
def visualization(df):
    fig = plt.figure(figsize = (24,10), dpi = 60)
    gs = GridSpec(ncols=10, nrows=12, left=0.05, right=0.5, wspace=0.2, hspace=0.1)
    fig.patch.set_facecolor('#f5f5f5')
    sns.set_palette(sns.color_palette(['#00f5d4','#f15bb5']))

    ax1 = fig.add_subplot(gs[1:6, 0:4])
    ax2 = fig.add_subplot(gs[8:, 0:4])
    ax3 = fig.add_subplot(gs[1:6, 5:])
    ax4 = fig.add_subplot(gs[8:, 5:])

    # axes list
    axes = [ ax1,ax2, ax3, ax4]

    # setting of axes; visibility of axes and spines turn off
    for ax in axes:
        ax.axes.get_yaxis().set_visible(False)
        ax.set_facecolor('#f5f5f5')
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)



    #-------Ax 1------------------------------------------------
    sns.kdeplot(x='avg_glucose_level', data=df[df.stroke==0], ax=ax1, shade=True, color='#00f5d4', alpha=1)
    sns.kdeplot(x='avg_glucose_level', data=df[df.stroke==1], ax=ax1, shade=True, color='#b30f72', alpha=0.8)
    ax1.set_xlabel('Average Glucose Level', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'black'})
    ax1.text(-20, 0.0175, 'Glucose Distribution by Stroke', {'font':'cursive', 'size':'20','color': 'black','weight':'bold'})

    ax1.text(200, 0.014,'Stroke', {'font':'cursive', 'fontsize':16, 'color':'#b30f72'})
    ax1.text(275, 0.014,'|', {'font':'Serif', 'fontsize':16,'fontweight':'bold', 'color':'black'})
    ax1.text(285, 0.014,'Healthy', {'font':'cursive', 'fontsize':16, 'color':'#00c5a4'})

    #-------Ax 2------------------------------------------------
    sns.kdeplot(x='age', data=df[df.stroke==0], ax=ax2, shade=True, 
            color='#00f5d4', alpha=1)
    sns.kdeplot(x='age', data=df[df.stroke==1], ax=ax2, shade=True, 
            color='#b30f72', alpha=0.8)
    ax2.set_xlabel('Age', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'black'})
    ax2.text(-20, 0.049, 'Age Distribution by Stroke', {'font':'cursive', 'size':'20','color': 'black','weight':'bold'})

    ax2.text(50, 0.042,'Stroke', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'#b30f72'})
    ax2.text(75, 0.042,'|', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'black'})
    ax2.text(80, 0.042,'Healthy', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'#00c5a4'})

    #-------Ax 3------------------------------------------------
    sns.kdeplot(x='bmi', data=df[df.stroke==0], ax=ax3, shade=True, 
            color='#00f5d4', alpha=1)
    sns.kdeplot(x='bmi', data=df[df.stroke==1], ax=ax3, shade=True, 
            color='#b30f72', alpha=0.8)
    ax3.set_xlabel('bmi', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'black'})
    ax3.text(0, 0.099, 'Bmi Distribution by Stroke', {'font':'cursive', 'size':'20','color': 'black','weight':'bold'})

    ax3.text(45, 0.062,'Stroke', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'#b30f72'})
    ax3.text(60, 0.062,'|', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'black'})
    ax3.text(63, 0.062,'Healthy', {'font':'cursive', 'fontsize':16,'fontweight':'bold', 'color':'#00c5a4'})


    ax4.text(0, 0.999, 'People with a BMI level of 30 is considered "obese"' +\
         ',and therefore,\nthe risk of stroke is quite high' +\
         'As seen on the graph, the highest \ndensity point is observed at the BMI level of 25-30.', 
         {'font':'Serif', 'color': 'black', 'size':14})
    ax4.text(0, 0.699, 'From the graph, it can be observed that the risk of stroke is quite' +\
         '\nhigh in the age range of 60-80.', 
         {'font':'Serif', 'color': 'black', 'size':14})
    ax4.text(0, 0.199, 'When considering glucose levels, the risk of stroke is high at every \nlevel. ' +\
         'Therefore, it would not be appropriate to evaluate glucose \nlevels alone.'
         'However, in individuals with glucose levels of 200 or above, \nthe risk of stroke may be higher.', 
         {'font':'Serif', 'color': 'black', 'size':14})
    #---------------------------------------------------------------
    fig.text(0.12, 0.9, 'Strokes by Glucose, Age & Bmi Factors', {'font':'cursive', 'weight':'bold','color': 'black', 'size':28})
    plt.show()



# start normalization
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe



def normalization(df, num_cols):
    # new features
    df['bmi_cat'] = pd.cut(df['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
    df['age_cat'] = pd.cut(df['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
    df['glucose_cat'] = pd.cut(df['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])

    #Encoding and scaling
    le = LabelEncoder()
    #binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
    binary_cols =[]
    for col in df.columns:
        if(df[col].dtype not in [int, float]) and df[col].nunique()==2:
            binary_cols.append(col)



    for col in binary_cols:
        df = label_encoder(df, col)
    
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)

    #scaler = StandardScaler()
    global scaler1
    scaler1.fit(df[num_cols])
    df[num_cols] = scaler1.transform(df[num_cols])
    print(df.head())
    return df

# end normalization





def normalization_single_input(df):

    print(df)

    # gender
    df['gender_Male']=0
    df["gender_Other"]=0
    if(df.loc[0,'gender']=='Male'):
        df.loc[0,'gender_Male']=1
    elif(df.loc[0,'gender']=='Other'):
        df.loc[0,"gender_Other"]=1

    df = df.drop('gender', axis=1)
    

    # work_type
    df["work_type_Never_worked"] =0
    df["work_type_Private"] =0
    df["work_type_Self-employed"] =0
    df["work_type_children"] =0

    if(df.loc[0,"work_type"] == 'Never_worked'):
        df.loc[0, "work_type_Never_worked"] = 1
    elif(df.loc[0,"work_type"] == 'children'):
        df.loc[0, "work_type_children"] = 1
    elif(df.loc[0,"work_type"] == 'Self-employed'):
        df["work_type_Self-employed"] = 1
    elif(df.loc[0,"work_type"] == 'Private'):
        df["work_type_Private"] = 1

    df =df.drop("work_type", axis=1)
    # smoking_status

    df["smoking_status_formerly smoked"] =0
    df["smoking_status_never smoked"] = 0
    df["smoking_status_smokes"] =0

    if(df.loc[0,"smoking_status"] == 'smokes'):
        df.loc[0, "smoking_status_smokes"] = 1
    elif(df.loc[0,"smoking_status"] == 'never smoked'):
        df.loc[0, "smoking_status_never smoked"] = 1
    elif(df.loc[0,"smoking_status"] == 'formerly smoked'):
        df["smoking_status_formerly smoked"] = 1

    df = df.drop('smoking_status', axis=1)    


    # new features
    df['bmi_cat'] = pd.cut(df['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
    df['age_cat'] = pd.cut(df['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
    df['glucose_cat'] = pd.cut(df['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])


    df["bmi_cat_Ideal"] =0
    df["bmi_cat_Overweight"] =0
    df["bmi_cat_Obesity"] =0

    if(df.loc[0,"bmi_cat"] == 'Ideal'):
        df.loc[0, "bmi_cat_Ideal"] = 1
    elif(df.loc[0,"bmi_cat"] == 'Overweight'):
        df.loc[0, "bmi_cat_Overweight"] = 1
    elif(df.loc[0,"bmi_cat"] == 'Obesity'):
        df["bmi_cat_Obesity"] = 1

    df = df.drop('bmi_cat', axis=1)
    


    df["age_cat_Teens"] =0
    df["age_cat_Adults"] =0
    df["age_cat_Mid Adults"] =0
    df["age_cat_Elderly"] =0

    if(df.loc[0,"age_cat"] == 'Teens'):
        df.loc[0, "age_cat_Teens"] = 1
    elif(df.loc[0,"age_cat"] == 'Adults'):
        df.loc[0, "age_cat_Adults"] = 1
    elif(df.loc[0,"age_cat"] == 'Mid Adults'):
        df["age_cat_Mid Adults"] = 1
    elif(df.loc[0,"age_cat"] == 'Elderly'):
        df["age_cat_Elderly"] = 1

    df = df.drop('age_cat', axis=1)


    df["glucose_cat_Normal"] =0
    df["glucose_cat_High"] = 0
    df["glucose_cat_Very High"] =0

    if(df.loc[0,"glucose_cat"] == 'Normal'):
        df.loc[0, "glucose_cat_Normal"] = 1
    elif(df.loc[0,"glucose_cat"] == 'High'):
        df.loc[0, "glucose_cat_High"] = 1
    elif(df.loc[0,"glucose_cat"] == 'Very High'):
        df["glucose_cat_Very High"] = 1

    df = df.drop('glucose_cat', axis=1)
 
    

    # numerical columns
    num_cols=['age','avg_glucose_level', 'bmi']
    #scaler = StandardScaler()
    global scaler1
    df[num_cols] = scaler1.transform(df[num_cols])

    print(df.head())
    return df


def modeling(dataframe):
    y = dataframe["stroke"]
    X = dataframe.drop("stroke", axis=1)

    
    smote=SMOTE()
    x_smote,y_smote=smote.fit_resample(X,y)

    X_train,X_test, y_train,y_test=train_test_split(x_smote,y_smote,test_size=0.33,random_state=42)


    #sc= StandardScaler()
    global scaler2
    scaler2.fit(X_train)
    X_train = scaler2.transform(X_train) 
    X_test = scaler2.transform(X_test)

    LR=LogisticRegression()
    LR.fit(X_train,y_train)
    y_pred=LR.predict(X_test)
    class_report=classification_report(y_test,y_pred)
    print(class_report)
    return LR



def predict(mylist):
    # load model
    loaded_model = pickle.load(open(filename, 'rb'))
    dataframe=pd.DataFrame.from_dict(mylist)
    dataframe = normalization_single_input(dataframe)
    X = scaler2.transform(dataframe)
    y_pred=loaded_model.predict(X)
    print(y_pred)
    if(y_pred[0]==1):
        return "Positive"
    return "Negative"



def main():
    print("============================ Program v1.0.0 ==============================")
    current_time_str = datetime.datetime.now().strftime(time_string_format)
    print("Up Time:", current_time_str)

    # detect the running operating system. If operating system is unknown, then exit
    detect_system()
    if(system=='unknown'):
        print("Support unavailable. Sorry")
        return
    
    print("Detected Os: {}".format(system))
    print("pandas package version: {}".format(pd.__version__))
    print('')

    username = getpass.getuser()
    print(f"Hello {username}. Hope everything is fine. I am Stoke prediction model.\nHere These models can assist healthcare professionals in identifying individuals\nwho may be at high risk for stroke and implementing preventive measures.\n")
    
    
    # next target
    '''
        
    '''
    df = read_dataset()
    check_data(dataframe=df)
    df = df.drop('id', axis=1)
    cat_cols, num_cols, cat_but_car = identify_cols_types(df)


    # handling outlier
    handle_outliers(df, num_cols=num_cols)

    # handle missing values
    handle_missing_values(df)

    # data visualization
    #visualization(df)

    # normalize column data
    df = normalization(df, num_cols=num_cols)

    # modeling
    model = modeling(df)
    # save the model to a file
    
    pickle.dump(model, open(filename, 'wb'))
    
    print("\n\nModel is trained.\nLets predicate...")

    # rendering gui
    gui()



if __name__ == '__main__':
    main()
    print("End")