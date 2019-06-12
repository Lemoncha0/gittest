import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


fig = plt.figure()
fig.set(alpha=0.2)
data_train = pd.read_csv("train.csv")

plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title('Survival Situation')
plt.ylabel("number")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title('PClass')
plt.ylabel('number')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel('Age')
plt.grid(b=True, which='major', axis='y')
plt.title('Relationship between Age and Survival')

plt.subplot2grid((2, 3), (1, 0), colspan=2)     # 这里的参数colspan=3代表占2格（2,3）
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age distribution of passengers by class')
plt.legend(('first class', 'second class', 'third class'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('num of people at each embarkation')
plt.ylabel('number')

# 在各类等级的船舱中获救情况的分布
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'Saved': Survived_1, 'Not saved': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('Survival Situation of PClass')
plt.ylabel('number')
plt.xlabel(u'PClass')

# 获救与未获救的人当中男性与女性的分布
Survived_male = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_female = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'male': Survived_male, 'female': Survived_female})
df.plot(kind='bar', stacked=True)
plt.title('Survival Situation on Sex')
plt.ylabel('number')
plt.xlabel('Sex')

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'Saved': Survived_1, 'Not saved': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('Survival Situation on Embarked')
plt.ylabel('number')
plt.xlabel('Embarked')

fig2 = plt.figure()
fig.set(alpha=0.65)
plt.title("The Survival Rate of PClass and Sex ")


ax1 = fig2.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][
    data_train.Pclass != 3].value_counts().plot(kind='bar', label='female high class', color='#FA2479')
ax1.set_xticklabels(['Saved', 'not saved'], rotation=0)
ax1.legend(['female/high class '], loc='best')

ax2 = fig2.add_subplot(142)
data_train.Survived[data_train.Sex == 'female'][
    data_train.Pclass == 3].value_counts().plot(kind='bar', label='female high class')
ax2.set_xticklabels(['Saved', 'not saved'], rotation=0)
ax2.legend(['female/low class'], loc='best')

ax3 = fig2.add_subplot(143)
data_train.Survived[data_train.Sex == 'male'][
    data_train.Pclass != 3].value_counts().plot(kind='bar', label='male high class', color='yellow')
ax3.set_xticklabels(['Saved', 'not saved'], rotation=0)
ax3.legend(['male/high class'], loc='best')

ax4 = fig2.add_subplot(144)
data_train.Survived[data_train.Sex == 'male'][
    data_train.Pclass == 3].value_counts().plot(kind='bar', label='male high class', color='green')
ax4.set_xticklabels(['Saved', 'not saved'], rotation=0)
ax4.legend(['male/low class'], loc='best')

g1 = data_train.groupby(['SibSp', 'Survived'])
g2 = data_train.groupby(['Parch', 'Survived'])
df1 = pd.DataFrame(g1.count()['PassengerId'])
df2 = pd.DataFrame(g2.count()['PassengerId'])
print(df1, '\n', df2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_no_cabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({'Exist': Survived_cabin, 'Lack': Survived_no_cabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.show()

