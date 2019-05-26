import numpy as np

import matplotlib.pyplot as plt
def plots_for_nsources_range(title, names, scores, stds, n_samples):
    colors = ['b', 'g', 'r', 'orange', 'black']
    plt.figure()
    fig, ax = plt.subplots(figsize=(12,9))
    ax.grid(True)
    plt.title(title, fontsize=14)
    for i in range(len(scores)):
        ax.plot(n_samples, scores[i],'-o', label = names[i], color=colors[i])
        ax.fill_between(n_samples, scores[i] - stds[i], scores[i] + stds[i],alpha=0.10, color=colors[i])
        ax.set_ylabel('F1')
        ax.legend(loc=(0.8,0.1), fontsize=12)
        ax.set_ylabel('between y1 and 0')
        
    plt.xlabel('number of training samples per class', fontsize=12)
    plt.ylabel('F1', fontsize=12)
    plt.savefig('logs/nsources_{}.png'.format(title))
    plt.show()

elmo= np.array([[i for i in range(4)] for j in range(3)],dtype=np.float)+1
elmo[0] = 0.1*elmo[0]+0.5
elmo[1] = 0.01*elmo[1]+0.05

infersent= np.array([[i for i in range(4)] for j in range(3)],dtype=np.float)+1
infersent[0] = np.power(0.1*infersent[0],0.5)
infersent[1] = 0.01*infersent[1]+0.05

unsent= np.array([[i for i in range(4)] for j in range(3)],dtype=np.float)+1
unsent[0] = np.power(0.1*unsent[0],2)
unsent[1] = 0.01*infersent[1]+0.05

names = ['elmo','unsent','infersent']
data =  [elmo , unsent , infersent]
scores = [d[0] for d in data]
stds = [d[1] for d in data]
ranges = data[0][2] #[1,2,3,4]
print ranges
plots_for_nsources_range('MSCOCO LogReg with Embeddings', names, scores, stds, ranges)
