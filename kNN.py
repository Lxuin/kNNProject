import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("KNNAlgorithmDataset.csv") #extract the data from the given datafile

#convert to numpy array
data_np = data.values #data.values[0]-[568] line for every data

np.random.shuffle(data_np) #numpy randomly shuffle the numpy array data set

y = data_np[:, 0] # extract the first column/diagnosis variable and save it in y
x = data_np[:, 1:] #slicing: all rows + columns from 1 to end. = remove first column from data set //result for y 


# First 3 Diagnosis. Y(diagnosis) with related X value
print("\nY:\n", y[0:3:1])
print("\nX:\n", x[0:3:1]) 

print("\nData:\n", data_np[:3, :]) #show the very first 3 rows

step1 = int(len(x) * 0.7) # first step 70% of the data set length and convert to integer so apprx = 398
step2 = int(len(x) * 0.9) # 20% step so 90% of the data set length
step3 = len(x) # last step, used in order to enhance the readability

y_train = y[0:step1:1] 
x_train = x[0:step1:1]
data_np_train = data_np[0:step1:1]

y_validation = y[step1:step2:1]
x_validation = x[step1:step2:1]
data_np_test = data_np[step1:step2:1]

y_test = y[step2:step3:1]
x_test = x[step2:step3:1]
data_np_test = data_np[step2:step3:1]

class kNN:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, k: int, d: str):
        """
        initialise the kNN constructor
        parameters
        - x_train(np.ndarray): x values of the training data
        - y_train(np.ndarray): y value of the training data
        - k(int): k neighbors
        - d(str): distance function (manhattan, euclidean, or chebyshev)
        """
        
        self.k = k 
        self.distance_function = self.distance_function(d) # set distance function
        self.mean = np.mean(x_train, axis=0) # calculate mean of training data
        self.std = np.std(x_train, axis=0) # calculate standard deviation of training data
        self.x_train = (x_train - self.mean) / self.std #normalize the data (x - mean)/std
        self.y_train = y_train

    def distance_function(self, d: str):
        """
        return the distance function based on the given input string
        parameters:
        - d (str): distance function (manhattan, euclidean, or chebyshev).
        """
        
        # manhattan distance
        if d == "manhattan":
            def manhattan_distance(x1, x2):
                return np.sum(np.abs(x1 - x2), axis=1)
            return manhattan_distance
        # euclidean distance
        elif d == "euclidean":
            def euclidean_distance(x1, x2):
                return np.sqrt(np.sum((x1 - x2)**2, axis=1))
            return euclidean_distance
        # chebyshev distance
        elif d == "chebyshev":
            def chebyshev_distance(x1, x2):
                return np.max(np.abs(x1 - x2), axis=1)
            return chebyshev_distance
        else:
            print("Choose from manhattan, euclidean or chebyschev")

    def predict(self, x: np.ndarray):
        """
        Takes the test data and predicts the label
        parameters:
        - x (np.ndarray): x values of the data
        return y_pred (np.ndarray) predicted labels
        """
        xnorm = (x - self.mean) / self.std #normalize the x_data (x - mean)/std
        # calculate distances between each test sample and all training samples

        distances = np.array([self.distance_function(x, self.x_train) for x in xnorm]) # calculate the distances between the normalized x values and the training sample

        kN_indexes = np.argsort(distances)[:, :self.k] #get the indexes of k nearest training samples

        kN_labels = np.array([self.y_train[i] for i in kN_indexes]) #get labels of the k nearest training samples

        y_pred = [] # create an empty array
        for labels in kN_labels: # iterate through each set of labels in kN_labels
            label_count = {} # create an empty dictionary
            for label in labels: # iterate thorugh every single label in lables
                if label in label_count: # if already present -> count it up
                    label_count[label] += 1
                else:                     # else set it to one
                    label_count[label] = 1
            label_predict = max(label_count, key=label_count.get) # find the label with the highest count
            y_pred.append(label_predict) # put the predicted label into the y_pred list
        return np.array(y_pred)

    def confusion_matrix(self, x: np.ndarray, y_true: np.ndarray):
        """
        calculate the confusion matrix for the kNN.
        parameters:
        - x (np.ndarray): x values of the training data
        - y_true (np.ndarray): y value of the training data that are true
        return matrix (np.ndarray): confusion matrix
        """
        y_pred = self.predict(x) # use the predict function to predict the y values of the x data
        matrix = np.zeros([2,2]) # size of the confusion matrix (since we have y = 0 or 1 , it is 2x2)
        matrix[0, 0] = np.sum(((y_true == 0) & (np.array(y_pred) == 0))) # True Negative (benign actual, benign predicted)
        matrix[0, 1] = np.sum(((y_true == 0) & (np.array(y_pred) == 1))) # False Negative (benign actual, malign predicted)
        matrix[1, 0] = np.sum(((y_true == 1) & (np.array(y_pred) == 0))) # False Positive (malign actual, benign predicted)
        matrix[1, 1] = np.sum(((y_true == 1) & (np.array(y_pred) == 1))) # True Positive (malign actual, malign predicted)
        return matrix
    
k_values = [1,3,5,7,9,11,13,15] #random number of values for the neighbors
d_values = ["manhattan", "euclidean", "chebyshev"] #array of the d distance function parameter (used to iterate over later on)

# initiliaze the needed variables in order to return them later
bestAccu = 0.0
best_k = 0
best_d = ""

for k in k_values: #iterate over the k values. start with 1 for example
    for d in d_values: #iterate over the distance functions with every single value in k_values
        knn_model = kNN(x_train, y_train, k, d) #create an object of kNN with the tariningsdata x + y
        x_valinorm = (x_validation - knn_model.mean) / knn_model.std #normalize the validation data that we set up in the 2nd task

        distances = []
        for x in x_valinorm: #iterate over the normailzed validation data and get the distances between each validuation sample and all the training samples
            distance = knn_model.distance_function(x, knn_model.x_train)
            distances.append(distance)
        distances = np.array(distances)

        kN_indexes = np.argsort(distances)[:, :k] #grabbing the indexes of the closest k sample for each validation sample

        kN_labels = []
        for i in kN_indexes:  #receive the labels of the closest k training samplest
            label = knn_model.y_train[i]
            kN_labels.append(label)
        kN_labels = np.array(kN_labels, dtype=int)

        y_predivali = []
        for labels in kN_labels:# assume the label for each validation sample by iteration over kN_labels //bincount()=count values of integer approaches. example: labels = [0,1,1,2,2,2] then: np.bincount(labels)=[1,2,3] //argmax() = returns the index of the highest number in the array
            predicted_label = np.bincount(labels).argmax()
            y_predivali.append(predicted_label)
        
        accuracy = np.sum(y_validation == y_predivali) / len(y_validation) #calculate the accuracy for the validation
        
        # in case the current accuracy is higher than bestaccuracy then we refresh our variables for it
        if accuracy > bestAccu:
            bestAccu = accuracy
            best_k = k
            best_d = d


best_knn_model = kNN(x_train, y_train, best_k, best_d) # set up a new object of knn but this with the best k and d value 
x_tempnorm = (x_test - best_knn_model.mean) / best_knn_model.std #normalize the test data for x we initialized in task 2

distances_temp = []
for x in x_tempnorm:  #calculate the distances between each test sample and all the training samples by iterataating over x_tempnorm
    distance = best_knn_model.distance_function(x, best_knn_model.x_train)
    distances_temp.append(distance)
distances_temp = np.array(distances_temp)

kN_indexes_test = np.argsort(distances_temp)[:, :best_k] # get the indxes of the closest k training sample for each test sample

kN_labels_test = []
for i in kN_indexes_test: #get the labels for the k closest trainings sample by iteration over k_inexes_test. dtype = int set all the datatypes in the nump array to integer
    label = best_knn_model.y_train[i]
    kN_labels_test.append(label)
kN_labels_test = np.array(kN_labels_test, dtype=int)

y_pred_test = []
for labels in kN_labels_test: #predict the label for each test sample
    predicted_label = np.bincount(labels).argmax()
    y_pred_test.append(predicted_label)

tempaccu = np.sum(y_test == y_pred_test) / len(y_test)# calcuate the accuracy for tempaccu
confusionMatr = best_knn_model.confusion_matrix(x_test, y_test) # generate the confusion matrix for the test predictions


print("Best k-Value: ", best_k)
print("Best d-Value: ", best_d)
print("Best Accuracy: ", bestAccu)
print("Conf-Matrix:\n", confusionMatr)

plt.scatter(x_train[:, :2][:, 0], x_train[:, :2][:, 1], c="r", marker="o", label="Training Points") # red circles
plt.scatter(x_validation[:, :2][:, 0], x_validation[:, :2][:, 1], c="b", marker="s", label="Validation Points")  # blue squares

plt.legend() # shows a legend
plt.title("Visualization of Training and Validation Data")  # title for the plot
plt.show()