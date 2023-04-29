import project
import pandas as pd
import pytest


"""
@pytest.mark.parametrize("dataframe, expected", [
    ("",True),
    ("02-02-2023 00:00:00",False)
])"""
def test_identify_cols_types():
    data = {"calories": [420, 380, 390,500,100,200,300,150,450,1000,740,750,780,800],"duration": [40,78,99,15,20,50, 40, 45,1,2,30,100,4,50]}

    #load data into a DataFrame object:
    df = pd.DataFrame(data)

    assert project.identify_cols_types(dataframe=df) == ([],["calories","duration"],[])


def test_check_outlier():
    data = {
        "calories": [420, 380, 390,500,100,200,300,150,450,1000,740,750,780,800],
            "duration": [40,78,99,15,20,50, 40, 45,1,2,30,100,4,50]
            }

    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    assert project.check_outlier(dataframe=df,col_name="calories") == False