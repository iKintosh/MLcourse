import pandas as pd
import numpy as np

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
        'capital-loss', 'hours-per-week', 'native-country', 'salary']
data = pd.read_csv('adult.data', header=None, names=names)
# print(data.head())

# 1. How many men and women (sex feature) are represented in this dataset?
print('How many men and women (sex feature) are represented in this dataset?')
print(data['sex'].value_counts())

# 2. What is the average age (age feature) of women?
print('What is the average age (age feature) of women?')

data['sex'] = data['sex'].str.strip()
print(data.loc[data['sex'] == 'Female', 'age'].mean())

# 3. What is the percentage of German citizens (native-country feature)?
print('What is the percentage of German citizens (native-country feature)?')
print(data['native-country'].value_counts(normalize=True)[' Germany'])

# 4-5. What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature)
# and those who earn less than 50K per year?
print('What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature) \
and those who earn less than 50K per year?')
d = {' <=50K': False, ' >50K': True}
data['is_more_than_50'] = data['salary'].map(d)
for flag in data['is_more_than_50'].unique():
    print( flag, data[data['is_more_than_50'] == flag]['age'].std())

# second idea
print(data.pivot_table(['age'], ['salary'], aggfunc=['std', 'mean']))

# 6. Is it true that people who earn more than 50K have at least high school 
# education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)
print('Is it true that people who earn more than 50K have at least high school \
education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)')
ed=[' Bachelors', ' Prof-school', ' Assoc-acdm', ' Assoc-voc', ' Masters',' Doctorate']
for e in data[data['salary']==' >50K']['education'].unique():
    if e not in ed:
        print(False)
        break

# 7. Display age statistics for each race (race feature) and each gender (sex feature). 
# Use groupby() and describe(). Find the maximum age of men of Amer-Indian-Eskimo race.
print('Display age statistics for each race (race feature) and each gender (sex feature). \
Use groupby() and describe(). Find the maximum age of men of Amer-Indian-Eskimo race.')

print(data.groupby(['race', 'sex'])['age'].describe())

# 8. Among whom is the proportion of those who earn a lot (>50K) greater: married or single men 
# (marital-status feature)? Consider as married those who have a marital-status starting with Married 
# (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.
print(' Among whom is the proportion of those who earn a lot (>50K) greater: married or single men \
(marital-status feature)? Consider as married those who have a marital-status starting with Married \
(Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.')

d={' Married-civ-spouse': True, ' Married-spouse-absent': True, ' Married-AF-spouse': True}
data['is_married'] = data['marital-status'].map(d)
data.loc[pd.isna(data['is_married']), 'is_married'] = False
print(data[(data['sex']=='Male') & (data['salary']==' >50K')]['is_married'].value_counts(normalize=True))

# 9. What is the maximum number of hours a person works per week (hours-per-week feature)? 
# How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?
print('What is the maximum number of hours a person works per week (hours-per-week feature)? \
How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?')
max_hours = data['hours-per-week'].max()
print('max hours per week: ', max_hours)
print('num of people: ', len(data[data['hours-per-week'] == max_hours]))
print(data[data['hours-per-week'] == max_hours]['is_more_than_50'].value_counts(normalize=True))
print('HERE: ', data[data['hours-per-week'] == max_hours].shape[0])

# 10. Count the average time of work (hours-per-week) for those who earn a little and a lot (salary) 
# for each country (native-country). What will these be for Japan?
print('Count the average time of work (hours-per-week) for those who earn a little and a lot (salary)\
 for each country (native-country). What will these be for Japan?')
print(data.groupby(['native-country', 'salary'])['hours-per-week'].mean()[' Japan'])

print(data.pivot_table(['hours-per-week'], ['native-country', 'salary'], aggfunc='mean').loc[[' Japan']])