import  numpy as np
import pandas as pd
import sys
from day02.ex09.data_spliter import data_spliter
from day03.ex06.my_logistic_regression import MyLogisticRegression as Mlr
from utils.utils_function import Standardization
import matplotlib.pyplot as plt

USAGE = "usgage: python mono_log.py -zipcode=0/1/2/3"

def check_input():
    """
    check argument –zipcode=x with x being 0, 1, 2 or 3. If no argument, usage and exit
    will be displayed
    check the input argument has to be as expect 
    the expected input  is -zipcode=0/1/2/3
    return:
        zipecode : int
    """
    if len(sys.argv) != 2 :
        print(USAGE)
        sys.exit(1)
    parameters = sys.argv[1].split('=')
    if len(parameters) != 2 or parameters[0] != "-zipcode" or parameters[1] not in  ("0", "1", "2", "3") :
        print(USAGE)
        sys.exit(1)
    return int(parameters[1])


def binarize(label:np.ndarray, zipcode:int|float, callback):
    """
        Select Space Zipcode or Threshold and generate a new numpy.array 'label_binary ' to label each
        citizen according the selection criterion:
        • 1 if the citizen’s zipcode corresponds to your favorite planet.
        • 0 if the citizen has another zipcode.
        The function then uses np.where to identify the indices where the label array is equal to the given zipcode.
        arguments: 
            label : is a numpy array representing the label data
            zipcode or Threshold: is an integer. 
            callback: function to realise the condition
        return :
            label_binary
    """
    #create label_binary that has the same shape as the input label array and initializes it with all zeros.
    label_binary = np.zeros(label.shape)
    label_binary[ callback(label, zipcode) ] = 1
    return label_binary



def draw(ax,  x, true_value, pred, zipecode:int, biometric_info):
    colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
    ax.scatter(x, true_value, c=colors[zipecode], label='true value')
    ax.scatter(x, pred, c=colors[zipecode + 3], label='pred value')

    ax.set_xlabel(biometric_info)
    ax.set_ylabel("Cityzen")

    ax.legend()
    ax.grid()


feature = pd.read_csv('solar_system_census.csv', index_col=0).to_numpy()
label = pd.read_csv('solar_system_census_planets.csv', index_col=0).to_numpy()

# label = label.to_numpy()
# feature = feature.to_numpy()

#check the valide argument  
#1. Take an argument: –zipcode=x with x being 0, 1, 2 or 3. 
zipcode = check_input()

#take Zipcode and generate a new array 'label_binary' composed of 0 or 1
label_binary = binarize(label, zipcode, lambda a, b: np.where(a == b))

#2. Split the dataset 
print(f"label = {label.shape}")
print(f"feature = {feature.shape}")

# print(f"x_ = {feature.value.shape}")


x_train, x_test, y_train, y_test = data_spliter(feature, label_binary, 0.2)


#3 normalization on your datase beacause it from difference bases
feature_standardizer = Standardization()
feature_standardizer.fit(x_train)
x_train_std = feature_standardizer.transform(x_train)
x_test_std = feature_standardizer.transform(x_test)

#4.train a logistic model to predict if a citizen comes from your favorite planet or not, using your brand new label.
theta = np.random.rand(feature.shape[1] + 1, 1)
model = Mlr(theta, 1e-2, max_iter=10000)
model.fit_(x_train_std, y_train)
pred = model.predict_(x_test_std)
pred_binary = binarize(pred, 0.5, lambda a, b: np.where(a >= b))


#Calculate and display the fraction of correct predictions over the total number of predictions based on the test set
accuracy = np.sum(pred_binary == y_test)/len(pred_binary)
print(f"Accuracy : {accuracy}")


#6. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(20, 6))
planete = ["Venus", "Earth", "Mars", "Asteroids’ Belt colonies"]
plt.suptitle("Planete " + planete[zipcode], fontsize=38, y=1)


draw(ax1, x_test[:, 0], y_test, pred_binary, zipcode, "Height")
draw(ax2, x_test[:, 1], y_test, pred_binary, zipcode, "Weight")
draw(ax3, x_test[:, 2], y_test, pred_binary, zipcode, "Bones Density")



plt.show()
