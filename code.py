import numpy, json, argparse
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score,f1_score, classification_report,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier
numpy.random.seed(1337)
import numpy as np
import pandas as pd



# Read in the NE data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
	print('Reading in data from {0}...'.format(corpus_file))
	words = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split()
			words.append(parts[0])
			if binary_classes:
				if parts[1] in ['GPE', 'LOC']:
					labels.append('LOCATION')
				else:
					labels.append('NON-LOCATION')
			else:
				labels.append(parts[1])	
	print('Done!')
	return words, labels

# Read in word embeddings 
def read_embeddings(embeddings_file):
	print('Reading in embeddings from {0}...'.format(embeddings_file))
	embeddings = json.load(open(embeddings_file, 'r'))
	embeddings = {word:numpy.array(embeddings[word]) for word in embeddings}
	print('Done!')
	return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
	vectorized_words = []
	for word in words:
		try:
			vectorized_words.append(embeddings[word.lower()])
		except KeyError:
			vectorized_words.append(embeddings['UNK'])
	return numpy.array(vectorized_words)
   
   
def runClassifier(epochs,batch_size,lr,kernel,X,Xtrain,Ytrain,Ytest,Xtest):
    numpy.random.seed(1337)                        
    # Define the properties of the perceptron model
    model = Sequential()
    model.add(Dense(input_dim = X.shape[1], units = Y.shape[1]))
    model.add(Activation(kernel))
    sgd = SGD(lr = lr)
    loss_function = 'mean_squared_error'
    model.compile(loss = loss_function, optimizer = sgd, metrics=['accuracy'])
    # Train the perceptron
    model.fit(Xtrain, Ytrain, verbose = 0, epochs = epochs, batch_size = batch_size)
    # Get predictions
    Yguess = model.predict(Xtest)
    # Convert to numerical labels to get scores with sklearn in 6-way setting
                       
    Yguess = numpy.argmax(Yguess, axis = 1)
    
    if args.binary:
        filename = "binary"
    else:
        filename = "multiclass"
    with open('bestParameters_'+filename+'.txt', 'a+') as file_object:
        str2 = str(epochs)+","+str(batch_size)+","+str(lr)+","+str(kernel)+","+str(f1_score(Ytest, Yguess,average='micro'))+","+str(f1_score(Ytest, Yguess,average='macro'))
        file_object.write(str2+"\n")
        print(str2)
   
    return Yguess
   
if __name__ == '__main__':

    # *** 3.2.2 Perceptron Hyperparameters ***
    findBestParameters = True
    findGeneralization = False

    parser = argparse.ArgumentParser(description='KerasNN parameters')
    parser.add_argument('data', metavar='named_entity_data.txt', type=str, help='File containing named entity data.')
    parser.add_argument('embeddings', metavar='embeddings.json', type=str, help='File containing json-embeddings.')
    parser.add_argument('-b', '--binary', action='store_true', help='Use binary classes.')
    args = parser.parse_args()
    # Read in the data and embeddings
    X, Y = read_corpus(args.data, binary_classes = args.binary)
    embeddings = read_embeddings(args.embeddings)



    # 3.2.3 Word Embeddings and Generalization
    split_point = int(0.75*len(X))
    Xtrain_ = X[:split_point]
    Xtest_ = X[split_point:]
    difference_X = list(set(Xtest_) - set(Xtrain_))
    indexes = [X.index(a)-split_point for a in difference_X]

    indexes = [3279,5997,8822]


    #print(difference_X[0])
    #print(Xtest_[indexes[0]])
                
    X = vectorizer(X, embeddings)
    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(Y) # Use encoder.classes_ to find mapping of one-hot indices to string labels
    if args.binary:
        Y = numpy.where(Y == 1, [0,1], [1,0])
    
    
    # Split in training and test data
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]
    Ytest = numpy.argmax(Ytest, axis = 1)
    

    
    print("epochs,batch_size,lr,kernel,f1_score(micro),f1_score(macro)")
    if findBestParameters:
        for epochs in range(1,10):
            for batch_size in [2**x for x in range(2,8)]:
                for lr in [10**x for x in range(-6,0)]:
                    for kernel in ['linear','relu']:
                        runClassifier(epochs,batch_size,lr,kernel,X,Xtrain,Ytrain,Ytest,Xtest)
    else:

        Yguess = runClassifier(1,32,.01,"linear",X,Xtrain,Ytrain,Ytest,Xtest)
    
        # 3.2.4 Error analysis        
        print("\n\n *** Confusion matrix ***")
        cm = confusion_matrix(Ytest, Yguess,normalize='true')
        cm = {encoder.classes_[i]: cm[:,i] for i in range(len(cm[0]))}
        cm = pd.DataFrame.from_dict(cm,orient='index', columns=encoder.classes_)
        print(cm)
    
        # 3.2.1 Baseline
        print("\n\n *** Classification report on model ***")
        
        print('Classification accuracy on test: {0}'.format(accuracy_score(Ytest, Yguess)))
        print(classification_report(Ytest, Yguess,target_names=encoder.classes_))
        
        print("\n\n *** Classification report on baseline (stratified) ***")
        
        dummy_clf = DummyClassifier(strategy="stratified")
        dummy_clf.fit(Xtrain, Ytrain)
        DummyClassifier(strategy='stratified')
        Yguess = dummy_clf.predict(Xtest)
        Yguess = [np.argmax(a) for a in Yguess]   
        
        print(classification_report(Ytest, Yguess,target_names=encoder.classes_))
        
        
        # 3.2.3 Word Embeddings and Generalization
        if findGeneralization:
            print("Words which are in the test set but not in the training set")
            print(tuple(zip(indexes,difference_X,[encoder.classes_[a] for a in Ytest[indexes]])))

            print("Words which are in the training set")
            print(tuple(zip(Xtrain_,[encoder.classes_[np.argmax(a)] for a in Ytrain])))
            
            print("Results on only words which are in the test set but not in the training set")
            Yguess = runClassifier(1,32,.01,"linear",X,Xtrain,Ytrain,Ytest[indexes], Xtest[indexes])
            
            print("Yguess")
            print(encoder.classes_[Yguess])
            
            print("GT")
            print(encoder.classes_[Ytest[indexes]])
    
    
    
