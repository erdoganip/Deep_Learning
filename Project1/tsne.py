##İpek Erdoğan
##2019700174
##Deep Learning HW1

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

def tsne_plot(filename,word_dictionary,one_hot_dictionary):
    labels = []
    embeddings = []
    file = open(filename, 'rb')
    weightdict = pickle.load(file)
    w1 = weightdict["w1"]

    for keys in one_hot_dictionary:
        embedding = np.matmul(one_hot_dictionary[keys],w1)
        embeddings.append(embedding)
        labels.append(word_dictionary[keys])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(embeddings)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()