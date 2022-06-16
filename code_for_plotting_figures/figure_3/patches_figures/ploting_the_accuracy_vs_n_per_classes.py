# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:23:54 2022

@author: gligorov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#maybe I should have just exported the matrices instead of doing this manual work of creating excels

#loading overall results
df = pd.read_excel (r"averaged_across_classes.xlsx")
print(df)

runs = 3

accuracy_overall = np.zeros((3,10))
for i in range(runs):
    for n in range(1,11):
        accuracy_overall[i,n-1] = float(df.iloc[i][n])
        
print(accuracy_overall)

#loading DAPT results
df = pd.read_excel(r'DAPT.xlsx')


runs = 3

accuracy_DAPT = np.zeros((3,10))
for i in range(runs):
    for n in range(1,11):
        accuracy_DAPT[i,n-1] = float(df.iloc[i][n])
        
#loading tbx6 results
df = pd.read_excel(r'tbx6.xlsx')


runs = 3

accuracy_tbx6 = np.zeros((3,10))
for i in range(runs):
    for n in range(1,11):
        accuracy_tbx6[i,n-1] = float(df.iloc[i][n])
        
#loading her1;her7 results
df = pd.read_excel(r'her1_her7.xlsx')


runs = 3

accuracy_h1h7 = np.zeros((3,10))
for i in range(runs):
    for n in range(1,11):
        accuracy_h1h7[i,n-1] = float(df.iloc[i][n])
        
#loading WT results
df = pd.read_excel(r'WT.xlsx')


runs = 3

accuracy_WT = np.zeros((3,10))
for i in range(runs):
    for n in range(1,11):
        accuracy_WT[i,n-1] = float(df.iloc[i][n])
        
        
#plotting
        
x_axis = range(1,11)

for i in range(runs):
    plt.scatter(x_axis, accuracy_overall[i,:]*100)
plt.title("Overall accuracy")
plt.xlabel("N")
plt.ylabel("Accuracy [%]")
plt.show()

for i in range(runs):
    plt.scatter(x_axis, accuracy_tbx6[i,:]*100)
plt.title("tbx6")
plt.xlabel("N")
plt.ylabel("Accuracy [%]")
plt.show()

for i in range(runs):
    plt.scatter(x_axis, accuracy_h1h7[i,:]*100)
plt.title("her1;her7")
plt.xlabel("N")
plt.ylabel("Accuracy [%]")
plt.show()

for i in range(runs):
    plt.scatter(x_axis, accuracy_DAPT[i,:]*100)
plt.title("DAPT")
plt.xlabel("N")
plt.ylabel("Accuracy [%]")
plt.show()

for i in range(runs):
    plt.scatter(x_axis, accuracy_WT[i,:]*100)
plt.title("WT")
plt.xlabel("N")
plt.ylabel("Accuracy [%]")
plt.show()
