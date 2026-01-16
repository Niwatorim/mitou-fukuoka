import csv
import os,sys

csvs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "csv_s")

import pandas as pd
def param_testing(csv_name:str):
    path = os.path.join(csvs_path,csv_name)
    df = pd.read_csv(path)
    for i in range(df.shape[0]):
        pass

"""
User writes CSV file and gets the content,
writes file_name in there, if already there, give warning first
send that in Langgraph self. param testing values check and send to the AI, based on titles. Make titles the same as the ones on the content ur tryna fill out
send that to all prompts

user makes csv file with all the values -> saved as the exact csv they want

AI makes code, and then we add with hard coding the for loop

then in param testing tab, they can check the file name of the csv -> runs the entire code for them and returns the results (print logs)

csv file:
email password expected_response_email expected_response_password



file:

HARDCODED PART
import csv
path = user_defined_name.py

with open(path,w) as f:
    config = csv file
dictionary: bla bla

AI GENERATED PART
for i in range (rows):
    playwright script
    input values config[f]





"""





