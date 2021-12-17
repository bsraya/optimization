from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

iris = datasets.load_iris()
features = iris.data 
target = iris.target

sepal_length = np.array(features[:, 0])
petal_width = np.array(features[:, 3])

species_names = list()

for i in target:
    if i == 0:
        species_names.append('setosa')
    elif i == 1:
        species_names.append('versicolor')
    else:
        species_names.append('virginica')

def predict_list(intercept, coefficient, dataset):
    return np.array([intercept + coefficient * x for x in dataset])

def gd(x, y, epochs, df, alpha = 0.01):
    length = len(x)
    intercept, coefficient = 0.0, 0.0
    for epoch in range(epochs):
        sum_error = 0.0
        predictions = predict_list(intercept, coefficient, x)
        b0_error = (1/length) * np.sum(predictions - y)
        b1_error = (1/length) * np.sum((predictions - y) * x)
        intercept = intercept - alpha * b0_error
        coefficient = coefficient - alpha * b1_error 
        sum_error = sum_error + np.sum((predictions - y) ** 2) / (2 * length)
        df.loc[epoch] = [intercept, coefficient, sum_error]
    return df

gd_loss = pd.DataFrame(columns=['intercept', 'coefficient', 'sum_error'])
gd_loss = gd(sepal_length, petal_width, epochs = 10000, df = gd_loss)

fig, ax = plt.subplots(figsize=(10,5))

ax.set_xlim(4.0, 8.0)
ax.set_ylim(0.0, 3.0)

sns.scatterplot(
    x = sepal_length, 
    y = petal_width, 
    hue = species_names
)

line, = ax.plot(sepal_length, gd_loss['intercept'][0] + gd_loss['coefficient'][0] * sepal_length,'r-', linewidth = 2)
def update(frame_num):
    label = 'timestep {0}'.format(frame_num + 1)
    line.set_ydata(
        gd_loss['intercept'].loc[frame_num] + gd_loss['coefficient'].loc[frame_num] * sepal_length
    )
    ax.set_xlabel(label)
    return line, ax

anim = FuncAnimation(fig, update, repeat = True, frames=np.arange(0, 10000, 50), interval=10)
anim.save('./animation.gif', writer='imagemagick', fps=60)