'''
Project: Mini Project 1 
Authors: Everyone write your names, Jacob Harper
Group ID: 15

'''

import os
import numpy as np


#######################################  adult  #######################################

#Instantiate array and add headings row
adult = np.empty((0,15))
headings = np.array([['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']])
adult = np.append(adult, headings.reshape(1,15), axis=0)


#Opening the file
script_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_directory, 'adult.data')
fadult = open(filename, 'r')

#Copying all data into matrix
for line in fadult:
    values = line.strip().split(', ')
    row = np.array([[]])
    for value in values:
        if '.' in value:
            row = np.append(row, float(value))  # Convert to float if it contains a decimal point
        elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
            row = np.append(row, int(value))  # Convert to int if it's a valid integer
        else:
            row = np.append(row, value)  # Keep as string if it's neither float nor int

    adult = np.append(adult, row.reshape(1,15), axis=0)

fadult.close()


#np.save('adult_num', adult, allow_pickle=True, fix_imports=True)

#######################################  car  #######################################

#Instantiate array and add headings row
car = np.empty((0,7))
headings = np.array([['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety', 'class']])
car = np.append(car, headings.reshape(1,7), axis=0)


#Opening the file
script_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_directory, 'car.data')
fcar = open(filename, 'r')

#Copying all data into matrix
for line in fcar:
    values = line.strip().split(',')
    row = np.array([[]])
    for value in values:
        if '.' in value:
            row = np.append(row, float(value))  # Convert to float if it contains a decimal point
        elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
            row = np.append(row, int(value))  # Convert to int if it's a valid integer
        else:
            row = np.append(row, value)  # Keep as string if it's neither float nor int

    car = np.append(car, row.reshape(1,7), axis=0)

fcar.close()

#np.save('car_num', car, allow_pickle=True, fix_imports=True)

#######################################  ionosphere  #######################################

#Instantiate array and add headings row
ionosphere = np.empty((0,35))
headings = np.array([['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10', 'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15', 'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20', 'Attribute21', 'Attribute22', 'Attribute23', 'Attribute24', 'Attribute25', 'Attribute26', 'Attribute27', 'Attribute28', 'Attribute29', 'Attribute30', 'Attribute31', 'Attribute32', 'Attribute33', 'Attribute34', 'class']])
ionosphere = np.append(ionosphere, headings.reshape(1,35), axis=0)


#Opening the file
script_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_directory, 'ionosphere.data')
fionosphere = open(filename, 'r')

#Copying all data into matrix
for line in fionosphere:
    values = line.strip().split(',')
    row = np.array([[]])
    for value in values:
        if '.' in value:
            row = np.append(row, float(value))  # Convert to float if it contains a decimal point
        elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
            row = np.append(row, int(value))  # Convert to int if it's a valid integer
        else:
            row = np.append(row, value)  # Keep as string if it's neither float nor int

    ionosphere = np.append(ionosphere, row.reshape(1,35), axis=0)


fionosphere.close()

#np.save('ionosphere_num', ionosphere, allow_pickle=True, fix_imports=True)

#######################################  iris  #######################################

#Instantiate array and add headings row
iris = np.empty((0,5))
headings = np.array([['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']])
iris = np.append(iris, headings.reshape(1,5), axis=0)


#Opening the file
script_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_directory, 'iris.data')
firis = open(filename, 'r')

#Copying all data into matrix
for line in firis:
    values = line.strip().split(',')
    row = np.array([[]])
    for value in values:
        if '.' in value:
            row = np.append(row, float(value))  # Convert to float if it contains a decimal point
        elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
            row = np.append(row, int(value))  # Convert to int if it's a valid integer
        else:
            row = np.append(row, value)  # Keep as string if it's neither float nor int

    iris = np.append(iris, row.reshape(1,5), axis=0)

firis.close()

#np.save('iris_num', iris, allow_pickle=True, fix_imports=True)
