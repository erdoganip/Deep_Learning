##İpek Erdoğan
##2019700174
##Deep Learning HW1
import numpy as np
import time
import pickle

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def deriv_sigmoid(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def softmax(x):
    z = np.exp(x)
    sum = np.sum(z)
    softmax = z/sum
    return softmax

def stable_softmax(x):
    z=[i-max(i) for i in x]
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax

def cross_entropy(prediction,target):
    length = len(prediction)
    soft_prediction = stable_softmax(prediction)
    loss = -np.sum(np.multiply(target,np.log(soft_prediction)) + np.multiply((1 - target),np.log(1 - soft_prediction)) )/length
    return loss , soft_prediction
def deriv_loss(prediction,target):
    return np.subtract(prediction,target)


class Network(object):
    def __init__(self,onehot_dictionary,learning_rate):
        mean = 0
        standard_deviation = 0.5
        rows = 250
        columns = 16
        self.w1 = np.random.normal(mean, standard_deviation, (rows,columns))
        self.w2_1 = np.random.normal(mean, standard_deviation, (16, 128))
        self.w2_2 = np.random.normal(mean, standard_deviation, (16, 128))
        self.w2_3 = np.random.normal(mean, standard_deviation, (16, 128))
        self.w2 = np.concatenate((self.w2_1, self.w2_2, self.w2_3), axis=0)
        self.b1 = np.random.normal(mean, standard_deviation, 128)
        self.w3 = np.random.normal(mean, standard_deviation, (128,250))
        self.b2 = np.random.normal(mean, standard_deviation, 250)
        self.onehot_dictionary = onehot_dictionary
        self.learning_rate=learning_rate

    def save_weights(self,epoch_n):
        weightdict = {
            "w1": self.w1,
            "w2_1": self.w2_1,
            "w2_2": self.w2_2,
            "w2_3": self.w2_3,
            "w3": self.w3,
            "b1": self.b1,
            "b2": self.b2,
        }
        filename = "model{}.pk".format(epoch_n)
        file = open(filename, 'wb')
        pickle.dump(weightdict, file)

    def load_weights(self,filename):
        file = open(filename, 'rb')
        weightdict = pickle.load(file)
        self.w1 = weightdict["w1"]
        self.w2_1 = weightdict["w2_1"]
        self.w2_2 = weightdict["w2_2"]
        self.w2_3 = weightdict["w2_3"]
        self.w2 = np.concatenate((self.w2_1, self.w2_2, self.w2_3), axis=0)
        self.b1 = weightdict["b1"]
        self.w3 = weightdict["w3"]
        self.b2 = weightdict["b2"]

    def forward(self, batch, label):
        batch_losses=[]
        self.word1 = np.array([self.onehot_dictionary[batch[i][0]] for i in range(len(batch))]) #(64, 250)
        self.word2 = np.array([self.onehot_dictionary[batch[i][1]] for i in range(len(batch))]) #(64, 250)
        self.word3 = np.array([self.onehot_dictionary[batch[i][2]] for i in range(len(batch))]) #(64, 250)
        e1 = np.matmul(self.word1, self.w1) #w1.shape=(250, 16)
        e2 = np.matmul(self.word2, self.w1)
        e3 = np.matmul(self.word3, self.w1) #e3.shape=(64,16)
        self.e = np.concatenate((e1, e2, e3),axis=1) #e.shape=(64,48)
        self.h = np.matmul(self.e, self.w2) + self.b1 #w2.shape=(48, 128) h.shape= (64, 128)
        self.h_activated = sigmoid(self.h) #h_activated.shape=(64, 128)
        self.output = np.matmul(self.h_activated, self.w3) + self.b2 #output.shape=(64, 250)
        self.targets=np.array([self.onehot_dictionary[label[i]] for i in range(len(label))])
        loss,preds = cross_entropy(self.output,self.targets)
        batch_losses.append(loss)
        return sum(batch_losses)/len(batch_losses),preds

    def update_w1(self,common_w1_w2):
        #a=np.matmul(common_w1_w2,self.w2_1.T) #64X16
        w1_gradient_1 = np.matmul(self.word1.T,np.matmul(common_w1_w2,self.w2_1.T))
        w1_gradient_2 = np.matmul(self.word2.T,np.matmul(common_w1_w2,self.w2_2.T))
        w1_gradient_3 = np.matmul(self.word3.T,np.matmul(common_w1_w2,self.w2_3.T))
        w1_gradient= np.add(w1_gradient_1,w1_gradient_2,w1_gradient_3)
        self.w1=np.subtract(self.w1,np.multiply(w1_gradient,self.learning_rate))


    def update_w2(self,common_w1_w2):
        w2_gradient_total = np.matmul(self.e.T,common_w1_w2)
        w2_1_gradient = np.split(w2_gradient_total,3)[0]
        w2_2_gradient = np.split(w2_gradient_total,3)[1]
        w2_3_gradient = np.split(w2_gradient_total,3)[2]
        self.w2_1 = np.subtract(self.w2_1, np.multiply(w2_1_gradient, self.learning_rate))
        self.w2_2 = np.subtract(self.w2_2, np.multiply(w2_2_gradient, self.learning_rate))
        self.w2_3 = np.subtract(self.w2_3, np.multiply(w2_3_gradient, self.learning_rate))
        self.w2 = np.concatenate((self.w2_1, self.w2_2, self.w2_3), axis=0)

    def update_b1(self,common_w1_w2):
        b1_gradient = np.mean(common_w1_w2,axis=0)
        self.b1 = np.subtract(self.b1, np.multiply(b1_gradient, self.learning_rate))

    def update_w3(self):
        w3_gradient = np.matmul(self.h_activated.T,self.dLossdOutput)
        self.w3 = np.subtract(self.w3, np.multiply(w3_gradient, self.learning_rate))

    def update_b2(self):
        b2_gradient = np.mean(self.dLossdOutput,axis=0)
        self.b2 = np.subtract(self.b2, np.multiply(b2_gradient, self.learning_rate))

    def backward(self):
        self.dLossdOutput=deriv_loss(self.output,self.targets) #shape=64x250
        self.dSigmoid=deriv_sigmoid(self.h) #shape=64x128
        common_w1_w2=np.multiply(np.matmul(self.dLossdOutput,self.w3.T),self.dSigmoid) #shape=64x128
        self.update_w1(common_w1_w2)
        self.update_w2(common_w1_w2)
        self.update_b1(common_w1_w2)
        self.update_w3()
        self.update_b2()


