import  numpy as np
import pandas as pd
import sys
from day02.ex09.data_spliter import data_spliter
from day03.ex06.my_logistic_regression import MyLogisticRegression as Mlr
from utils.utils_function import Standardization
import matplotlib.pyplot as plt
import matplotlib as mpl
from mono_log import  binarize, draw



feature = pd.read_csv('solar_system_census.csv', index_col=0).to_numpy()
label = pd.read_csv('solar_system_census_planets.csv', index_col=0).to_numpy()

#set all label
label_b0 = binarize(label, 0, lambda a, b: np.where(a == b))
label_b1 = binarize(label, 1, lambda a, b: np.where(a == b))
label_b2 = binarize(label, 2, lambda a, b: np.where(a == b))
label_b3 = binarize(label, 3, lambda a, b: np.where(a == b))

x_train, x_test, y_train, y_test = data_spliter(feature, label, 0.2)
feature_standardizer = Standardization()
feature_standardizer.fit(x_train)
x_train_std = feature_standardizer.transform(x_train)
x_test_std = feature_standardizer.transform(x_test)

#train model 0
theta = np.random.rand(feature.shape[1] + 1, 1)
model_0 = Mlr(theta, 1e-3, 80000)
model_0.fit_(x_train_std, label_b0 )
pred_0 = model_0.predict_(x_test_std)


#train model 1
model_1 = Mlr(theta, 1e-3, 8000)
model_1.fit_(x_train_std, label_b1 )
pred_1 = model_1.predict_(x_test_std)

#train model 2
model_2 = Mlr(theta, 1e-3, 8000)
model_2.fit_(x_train_std, label_b2 )
pred_2 = model_2.predict_(x_test_std)

#train model 3
model_3 = Mlr(theta, 1e-3, 8000)
model_3.fit_(x_train_std, label_b3 )
pred_3 = model_3.predict_(x_test_std)

preds = np.hstack((pred_0, pred_1, pred_2, pred_3))
# print(pred_total_binary)
all_occuracy = np.argmax(preds, axis=1).reshape(-1, 1)
correct_pred = np.sum(all_occuracy == y_test) / y_test.shape[0]
print(f"Accuracy : {correct_pred:.4f}")


colors = ['#0066ff', '#00cc00', '#ff8c1a', '#ac00e6']
cmap = mpl.colors.ListedColormap(colors, name='from_list', N=None)
fig, axes = plt.subplots(1, 3, figsize=(13, 10))
# Fromating of scatter points for the expected and predicted
kws_expected = {'s': 300,
                'linewidth': 0.2,
                'alpha': 0.5,
                'marker': 'o',
                'facecolor': None,
                'edgecolor': 'face',
                'cmap': cmap,
                'c': y_test}
kws_predicted = {'s': 50,
                 'marker': 'o',
                 'cmap': cmap,
                 'c': all_occuracy}
axes[0].scatter(x_test[:, 0], x_test[:, 1],  label='expected', **kws_expected)
axes[1].scatter(x_test[:, 1], x_test[:, 2],  label='expected', **kws_expected)
axes[2].scatter(x_test[:, 2], x_test[:, 0],  label='expected', **kws_expected)

axes[0].scatter(x_test[:, 0], x_test[:, 1],  label='predicted', **kws_predicted)
axes[1].scatter(x_test[:, 1], x_test[:, 2],  label='predicted', **kws_predicted)
axes[2].scatter(x_test[:, 2], x_test[:, 0],  label='predicted', **kws_predicted)
scalarmapable = plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=0,
                                                           vmax=4),
                                      cmap)
cbar = fig.colorbar(scalarmapable,
                    orientation='horizontal',
                    label='Citizenship',
                    ticks=[0.5, 1.5, 2.5, 3.5],
                    ax=axes[:],
                    aspect=60, shrink=0.6)
cbar.ax.set_xticklabels(['Venus', 'Earth', 'Mars', 'Asteroids\nBelt'])
axes[0].set_xlabel("Height")
axes[0].set_ylabel("Weight")
axes[1].set_xlabel("Weight")
axes[1].set_ylabel("Bones")
axes[2].set_xlabel("Bones")
axes[2].set_ylabel("Height")
axes[0].legend(), axes[1].legend(), axes[2].legend()
axes[0].grid(), axes[1].grid(), axes[2].grid()
title = 'fraction of correct predictions = '
fig.suptitle(title + f'{correct_pred:0.4f}', fontsize=14)
plt.show()
