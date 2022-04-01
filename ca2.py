import numpy as np
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
#KMeans
class KMeans:
    def __init__(self, k):
        self.k = k
        self.centres = np.empty(self.k)
        self.old_centres = np.empty(self.k)
        self.distances = np.empty(self.k)
        self.labels = np.empty(self.k)
        self.convergence = np.empty(self.k)

    def random_representatives(self,data):
        random_representatives= [] 
        for i in range(len(data[0])):
            min = float(np.amin(data[:,i]))
            max = float(np.amax(data[:,i]))
            random_value = np.random.uniform(min,max)
            random_representatives.append(random_value)
        return np.asarray(random_representatives)

    def fit(self,data):
        #init arrays
        self.old_centres = np.zeros(len(self.centres))
        self.distances = np.zeros(len(self.centres))
        self.labels = np.zeros((len(classes)))
        self.convergence = np.zeros(len(self.centres))
        self.centres = np.zeros((self.k, data.shape[1]))
        
        # initial k representatives
        for i in range(self.k):
            self.centres[i] = self.random_representatives(data)
        
        #initial convergence value
        for i in range(len(self.convergence)):
            self.convergence[i] = l2_distance(self.old_centres[i],self.centres[i])
    
        while self.convergence.any() != 0: # loop till two centres doesnt change anymore
            #Assign each object to its closest representative
            for i in range(len(data)):
                for j in range(len(self.centres)):
                    self.distances[j] = l2_distance(data[i], self.centres[j])
                clusterid = np.argmin(self.distances)
                self.labels[i] = clusterid
            #update old_centres
            self.old_centres = np.copy(self.centres)
            #update new centres
            for i in range(self.k):
                class_object= []
                for j in range(len(self.labels)):
                    if self.labels[j] == i:
                        class_object.append(data[j])
                if len(class_object)!=0 :
                    self.centres[i] = np.mean(class_object,axis=0)
                
            #check convergence
            for i in range(len(self.convergence)):
                self.convergence[i] = l2_distance(self.centres[i], self.old_centres[i])
        return self.labels
#KMedians
class KMedians:
    def __init__(self, k):
        self.k = k
        self.centres = np.empty(self.k)
        self.old_centres = np.empty(self.k)
        self.distances = np.empty(self.k)
        self.labels = np.empty(self.k)
        self.convergence = np.empty(self.k)

    def random_representatives(self,data):
        random_representatives= [] 
        for i in range(len(data[0])):
            min = float(np.amin(data[:,i]))
            max = float(np.amax(data[:,i]))
            random_value = np.random.uniform(min,max)
            random_representatives.append(random_value)
        return np.asarray(random_representatives)

    def fit(self,data):
        #init arrays
        self.old_centres = np.zeros(len(self.centres))
        self.distances = np.zeros(len(self.centres))
        self.labels = np.zeros((len(classes)))
        self.convergence = np.zeros(len(self.centres))
        self.centres = np.zeros((self.k, data.shape[1]))
        
        # initial k representatives
        for i in range(self.k):
            self.centres[i] = self.random_representatives(data)
        
        #initial convergence value
        for i in range(len(self.convergence)):
            self.convergence[i] = l1_distance(self.old_centres[i],self.centres[i])
    
        while self.convergence.any() != 0: # loop till two centres doesnt change anymore
            #Assign each object to its closest representative
            for i in range(len(data)):
                for j in range(len(self.centres)):
                    self.distances[j] = l1_distance(data[i], self.centres[j])
                clusterid = np.argmin(self.distances)
                self.labels[i] = clusterid

            #update old_centres
            self.old_centres = np.copy(self.centres)

            #update new centres
            for i in range(self.k):
                class_object= []
                for j in range(len(self.labels)):
                    if self.labels[j] == i:
                        class_object.append(data[j])
                if len(class_object)!=0 :
                    self.centres[i] = np.median(class_object,axis=0)
                
            #check convergence
            for i in range(len(self.convergence)):
                self.convergence[i] = l1_distance(self.centres[i], self.old_centres[i])
        return self.labels

#Normalise data
def l2_norm(data):
    return data/(np.linalg.norm(data,axis=1,keepdims=True))

# Import Data function
def load_data(fname):
    objects = []
    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ') #strip space
            p.pop(0) #remove first word
            #add a new vaule to each objects to indicate the original label
            if fname == 'CA2data/animals':
                p.append(0)
            elif fname == 'CA2data/countries':
                p.append(1)
            elif fname == 'CA2data/fruits':
                p.append(2)
            elif fname == 'CA2data/veggies':
                p.append(3)
            p = [float(i) for i in p] # convert string to float
            objects.append(p)
    return np.array(objects)
# concatenate four datasets, create an non-labeled dataset 
# and its label array (used for computing the B-CUBED precision) and a normalized dataset
def make_data():
    a = load_data('CA2data/animals')
    c = load_data('CA2data/countries')
    f = load_data('CA2data/fruits')
    v = load_data('CA2data/veggies')
    dataset = np.concatenate((a,c,f,v),dtype=float)
    classes = np.copy(dataset[:,-1])
    dataset = dataset[:,:-1]
    norm = l2_norm(dataset)
    return dataset,classes,norm

def l2_distance(x,y):
    return np.linalg.norm(x - y)

def l1_distance(x,y):
    return np.sum(np.abs(x - y))

# B-CUBED
def bcubed(k,real_label,predict_label):
    labels = np.unique(real_label)
    precision_list= []
    recall_list = []
    fscore_list = []
    #for each clusters:
    for i in range(k):
        cluster =[]
        for j in range(len(predict_label)):
            if predict_label[j] == i:
                cluster.append(real_label[j])
        for l in labels: #going though all the labels
            counter=0
            # counter = np.count_nonzero(single_cluster==l)
            for x in range(len(cluster)):
                if (cluster[x]==l):
                    counter+=1 # Number of items in the cluster with label l 
            for _ in range(counter): #going though all subject with the same label
                precision = counter/len(cluster)
                precision_list.append(precision)
                recall = counter/np.count_nonzero(real_label==l)
                recall_list.append(recall)
                fscore_list.append((2 * precision * recall)/ (precision + recall))
    return np.mean(precision_list),np.mean(recall_list),np.mean(fscore_list)

def show_plt():
    #Graph Creation based off of model results
    plt.plot([k for k in range(1,10)],plot_scores[0][0:9], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_scores[1][0:9], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_scores[2][0:9], label = 'F-Score')
    
    plt.legend(loc="lower right")
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMeans')
    plt.savefig('Kmeans.jpg')
    plt.show()

    plt.plot([k for k in range(1,10)],plot_scores[0][9:], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_scores[1][9:], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_scores[2][9:], label = 'F-Score')
    plt.legend(loc="lower right")
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMedians')
    plt.savefig('Kmedians.jpg')
    plt.show()

    plt.plot([k for k in range(1,10)],plot_norm_scores[0][0:9], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_norm_scores[1][0:9], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_norm_scores[2][0:9], label = 'F-Score')
    plt.legend(loc="lower right")
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMeans (Normalised)')
    plt.savefig('Kmeans_Normalised.jpg')
    plt.show()

    # f_max_x=np.argmax(plot_norm_scores[2][9:])
    # f_max_y= plot_norm_scores[2][f_max_x]
    # f_max='Max Fscore: '+str(format(f_max_y, '.3f'))
    # plt.plot(f_max_x+1,f_max_y,'ko',color="red") 
    # plt.annotate(f_max,xy=(f_max_x,f_max_y),xytext=(f_max_x,f_max_y))
    plt.plot([k for k in range(1,10)],plot_norm_scores[0][9:], label = 'Precision')
    plt.plot([k for k in range(1,10)], plot_norm_scores[1][9:], label = 'Recall')
    plt.plot([k for k in range(1,10)], plot_norm_scores[2][9:], label = 'F-Score')
    plt.legend(loc="lower right")
    plt.xlabel('K')
    plt.ylabel('BCubed Score')
    plt.title('KMedians (Normalised)' )
    plt.savefig('Kmedians_Normalised.jpg')
    plt.show()


#make dataset, classes, and normalised data
data, classes, norm = make_data()
dataset  = np.copy(data)

# plot scores his
plot_scores = [[],[],[]]
plot_norm_scores = [[],[],[]]

#KMeans
for i in range(1,10): # k: 1-9
    kmeans = KMeans(k=i) #an instance of Kmeans
    kmeans.fit(data) # Clusteringbcubed
    precision,recall,fscore = bcubed(i,classes,kmeans.labels)
    # precision,recall,fscore = bscores(dataset,i,classes,kmeans.labels)
    plot_scores[0].append(precision)
    plot_scores[1].append(recall)
    plot_scores[2].append(fscore)
    print('KMeans K =',i)
    print('\nPrecision:',precision,'\nRecall:',recall,'\nFscore:',fscore,'\n')
    print("---")
    
    
#KMeans with Normalized data
for i in range(1,10): # k = 1-9
    kmeans = KMeans(k=i) 
    kmeans.fit(norm)
    precision,recall,fscore = bcubed(i,classes,kmeans.labels)
    # precision,recall,fscore = bscores(dataset,i,classes,kmeans.labels)
    plot_norm_scores[0].append(precision)
    plot_norm_scores[1].append(recall)
    plot_norm_scores[2].append(fscore)
    print('KMeans Normalized K =',i)
    print('\nPrecision:',precision,'\nRecall:',recall,'\nFscore:',fscore,'\n')
    print("---")
    
    
#KMedians
for i in range(1,10): # k: 1-9
    median = KMedians(k=i) 
    median.fit(data)
    precision,recall,fscore = bcubed(i,classes,median.labels)
    # precision,recall,fscore = bscores(dataset,i,classes,median.labels)
    plot_scores[0].append(precision)
    plot_scores[1].append(recall)
    plot_scores[2].append(fscore)
    print('Kmedians  K =',i)
    print('\nPrecision:',precision,'\nRecall:',recall,'\nFscore:',fscore,'\n')
    print("---")
    
    
#KMedians with Normalized data
for i in range(1,10): # k = 1-9
    median = KMedians(k=i) #an instance of Kmedians
    median.fit(norm) # Clustering
    precision,recall,fscore = bcubed(i,classes,median.labels)
    # precision,recall,fscore = bscores(dataset,i,classes,median.labels)
    plot_norm_scores[0].append(precision)
    plot_norm_scores[1].append(recall)
    plot_norm_scores[2].append(fscore)
    print('Kmedians Normalized  K =',i)
    print('\nPrecision:',precision,'\nRecall:',recall,'\nFscore:',fscore,'\n')
    print("---")
    
    



show_plt()