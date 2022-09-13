import numpy as np
import random
from sklearn import datasets
from sklearn import datasets
import matplotlib.pyplot as plt

INPUT_LAYER = 4
OUTPUT_LAYER = 3
HIDDEN_LAYER = 10
LEARNING_RATE = 0.001
EPOCHS = 100


iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
random.shuffle(dataset)
test = dataset[::5]
del dataset[::5]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    result = np.exp(x)
    return result/np.sum(result)

def cross_entropy(z, y):
    return -np.log(z[0, y])

def to_full(y, amount_classes):
    result = np.zeros((1, amount_classes))
    result[0, y] = 1
    return result

def relu_deriv(t):
    return (t>= 0).astype(float)

def predict(input):
    temp_1 = input @ weight_1 + bias_1
    h1 = relu(temp_1)
    temp_2 = temp_1 @ weight_2 + bias_2
    out = softmax(temp_2)
    return out

def calc_accuracy(test):
    correct = 0
    for x, y in test:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct/len(test)
    return acc


loss_arr = []
print('Do you want to teach network? 1/0')
toTeach = bool(input())

if toTeach:
    weight_1 = np.random.randn(INPUT_LAYER, HIDDEN_LAYER)
    bias_1 = np.random.randn(1, HIDDEN_LAYER)
    weight_2 = np.random.randn(HIDDEN_LAYER, OUTPUT_LAYER)
    bias_2 = np.random.randn(1, OUTPUT_LAYER)

    for ep in range(EPOCHS):
        random.shuffle(dataset)
        for i in range(len(dataset)):
            input, right_answer_index = dataset[i]
            # Forward prop
            temp_1 = input @ weight_1 + bias_1
            activated_lay1 = relu(temp_1)
            temp_2 = temp_1 @ weight_2 + bias_2
            out = softmax(temp_2)

            loss = cross_entropy(out, right_answer_index)

            # backward

            right_answer_full = to_full(right_answer_index, OUTPUT_LAYER)
            deriv_temp_2 = out - right_answer_full
            deriv_weight_2 = temp_1.T @ deriv_temp_2
            deriv_bias_2 = deriv_temp_2
            deriv_activated_lay1 = deriv_temp_2 @ weight_2.T
            deriv_temp_1 = deriv_activated_lay1 * relu_deriv(temp_1)
            deriv_weight_1 = input.T @ deriv_temp_1
            deriv_bias_1 = deriv_temp_1

            # update weights
            weight_1 = weight_1 - LEARNING_RATE * deriv_weight_1
            bias_1 = bias_1 - LEARNING_RATE * deriv_bias_1
            weight_2 = weight_2 - LEARNING_RATE * deriv_weight_2
            bias_2 = bias_2 - LEARNING_RATE * deriv_bias_2
            loss_arr.append(loss)
else:
    weight_1 = np.array([[0.44831817, -1.60243385, 1.20386644, 0.51088476, 0.49193183, -0.36233975, 0.5241643, -0.27336102, 0.80083547, 1.27665458],
                         [3.01883663, -0.39122725, -1.1287352, -0.67081027, 0.91542056, 1.16844094, -2.17963526, 0.55975669, -1.02556052, 0.3623952],
                         [0.89095443, 0.43430099, 0.09107906, -0.73211111, -0.59154428, -0.03242812, 0.24020295, 1.66684759, 1.58737041, 1.04473387],
                         [-1.61581353, 0.63957691, 0.05115084, -0.617744, -0.96916946, -1.20418053, -0.65919743, 1.65582481, 1.32522182, 2.30463141]])


    bias_1 = np.array([[-0.17112115,  1.26149943,  0.4432625,   0.29393664,  1.39490272,  0.35993034, 0.53210489, -0.05263293, 0.03753949, -0.85259899]])


    weight_2 = np.array([[-0.16038047,  0.13301062, -1.77921693],
                         [-0.1637214,   0.99710788,  0.25994604],
                         [-0.47833045,  0.13424318, -0.95015369],
                         [-1.34614721,  1.68800796,  1.79287918],
                         [ 0.58066696,  0.55769907, -1.4423919 ],
                         [ 0.35323853, -1.98512055, -0.14653566],
                         [ 0.68714914,  0.69252194,  0.06704794],
                         [-1.77432636, -1.4515969,  -0.34439878],
                         [-0.3749082,   0.62028159,  1.73298807],
                         [-0.26523415, -0.18621047,  0.2131783 ]])


    bias_2 = np.array([[ 1.77537923,  0.70447004, -0.28611826]])
    

#CREATE TESTS


acc = calc_accuracy(test)
print("Accuracy.....", acc)



#to update and save weights
#f = open('weights.txt', 'w')
# f.write('W1: ')
# f.write(str(weight_1))
# f.write('\n')
# f.write("b1: ")
# f.write(str(bias_1))
# f.write('\n')
# f.write('W2: ')
# f.write(str(weight_2))
# f.write('\n')
# f.write('B2: ')
# f.write(str(bias_2))
#f.close()


plt.plot(loss_arr)
plt.show()






















