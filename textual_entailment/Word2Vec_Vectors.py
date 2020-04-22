import numpy as np  # Make sure that numpy is imported
import keras
from scipy import spatial
from text_analysis.textual_entailment.KaggleWord2VecUtility import KaggleWord2VecUtility
from text_analysis.textual_entailment import Word2Vec_AverageVectors
from text_analysis.textual_entailment import rte_classify


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    # featureVec = np.array([], dtype="float32").reshape(num_features,)
    # featureVec = None
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    # print(len(words))
    for word in words:
        if word in index2word_set:
            # if featureVec is None:
            #     featureVec = model[word]
            # else:
            featureVec = np.vstack((featureVec, model[word]))

    return featureVec


def getCosFeatureVecs(text, hyp, model, num_features):
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array
    dataVecs = np.zeros((len(text),num_features),dtype="float32")
    maxLen = 0

    for (i,j) in zip(text, hyp):
        trainTextDataVecs = makeFeatureVec(i, model, num_features)
        trainHypDataVecs = makeFeatureVec(j, model, num_features)

        # featureVec = np.zeros((num_features,),dtype="float32")

        featureVec = np.array([spatial.distance.cosine(x, y) for x in trainTextDataVecs for y in trainHypDataVecs])

        # c = 0.
        # for x in trainTextDataVecs:
        #     for y in trainHypDataVecs:
        #         if x.shape and y.shape:
        #             featureVec[int(c)] = spatial.distance.cosine(x, y)

                # c = c + 1.

        featureVec = np.pad(featureVec, (0, num_features), 'constant', constant_values=(0, 0))
        featureVec = np.resize(featureVec, (num_features,))

        #
        # Print a status message every 1000th rte
        if counter % 1000. == 0.:
            print("Rte %d of %d" % (counter, len(text)))
        #
        # Call the function (defined above) that makes average feature vectors
        dataVecs[int(counter)] = featureVec

        #
        # Increment the counter
        counter = counter + 1.
    return dataVecs

def getMaxCosFeatureVecs(text, hyp, model, num_features):
    # Initialize a counter
    counter = 0.
    maxLen = 100
    #
    # Preallocate a 2D numpy array
    dataVecs = np.zeros((len(text),maxLen,num_features*2),dtype="float32")

    for (i,j) in zip(text, hyp):
        trainTextDataVecs = makeFeatureVec(i, model, num_features)
        trainHypDataVecs = makeFeatureVec(j, model, num_features)

        featureVec = np.zeros((num_features*2,),dtype="float32")

        # print(i, len(i), trainTextDataVecs.shape)
        # print(j, len(j), trainHypDataVecs.shape)

        if trainTextDataVecs.any() and trainHypDataVecs.any():
            for y, elem in enumerate(trainHypDataVecs):
                # featureVec = np.vstack((featureVec, elem))
                idx = np.argmax(np.array([spatial.distance.cosine(x, elem) for x in trainTextDataVecs]))
                # featureVec = np.vstack((featureVec, trainTextDataVecs[idx]))
                if y:
                #     print(y,j[y-1],idx,i[idx-1])
                # print(elem.shape)
                # print(trainTextDataVecs[idx].shape)
                    featureVec = np.vstack((featureVec, np.concatenate((elem, trainTextDataVecs[idx]))))
        else:
            featureVec = np.vstack((featureVec, np.zeros((1,num_features*2))))

        # print("featureVec ",featureVec.shape)
        featureVec = np.vstack((featureVec, np.zeros((maxLen-featureVec.shape[0],num_features*2),dtype="float32")))
        # print(featureVec.shape)
        # featureVec = np.pad(featureVec, (0, maxLen, num_features), 'constant', constant_values=(0, 0))
        # featureVec = np.resize(featureVec, (num_features,))

        #
        # Print a status message every 1000th rte
        if counter % 1000. == 0.:
            print("Rte %d of %d" % (counter, len(text)))
        #
        # Call the function (defined above) that makes average feature vectors
        dataVecs[int(counter)] = featureVec

        #
        # Increment the counter
        counter = counter + 1.
    return dataVecs


def rte_avgVecs(train, test, num_features, model, lstm=False):
    print("Creating average feature vecs for training reviews")

    train_text = [KaggleWord2VecUtility.review_to_wordlist(pair.text, True) for (pair, label) in train]
    train_hyp = [KaggleWord2VecUtility.review_to_wordlist(pair.hyp, True) for (pair, label) in train]
    train_label = np.array([[label] for (pair, label) in train])

    trainTextDataVecs = Word2Vec_AverageVectors.getAvgFeatureVecs(train_text, model, num_features)
    trainHypDataVecs = Word2Vec_AverageVectors.getAvgFeatureVecs(train_hyp, model, num_features)
    trainDataVecs = np.concatenate((trainTextDataVecs, trainHypDataVecs), axis=1)

    i = np.isnan(trainDataVecs)
    trainDataVecs[i] = 0

    print("Creating average feature vecs for test reviews")

    test_text = [KaggleWord2VecUtility.review_to_wordlist(pair.text, True) for (pair, label) in test]
    test_hyp = [KaggleWord2VecUtility.review_to_wordlist(pair.hyp, True) for (pair, label) in test]
    test_label = np.array([[label] for (pair, label) in test])

    testTextDataVecs = Word2Vec_AverageVectors.getAvgFeatureVecs(test_text, model, num_features)
    testHypDataVecs = Word2Vec_AverageVectors.getAvgFeatureVecs(test_hyp, model, num_features)
    testDataVecs = np.concatenate((testTextDataVecs, testHypDataVecs), axis=1)

    i = np.isnan(testDataVecs)
    testDataVecs[i] = 0

    if lstm:
        (m, n) = trainDataVecs.shape
        print(trainDataVecs.shape)
        trainDataVecs = np.reshape(trainDataVecs, (m, 2, int(n/2)))
        (m, n) = testDataVecs.shape
        testDataVecs = np.reshape(testDataVecs, (m, 2, int(n/2)))
        # exit()

        train_label = keras.utils.to_categorical(train_label)
        test_label = keras.utils.to_categorical(test_label)

    return (trainDataVecs, train_label, testDataVecs, test_label)


def rte_cosVecs(train, test, num_features, model, lstm=False):
    print("Creating feature vecs for training reviews")

    train_text = [KaggleWord2VecUtility.review_to_wordlist(pair.text, True) for (pair, label) in train]
    train_hyp = [KaggleWord2VecUtility.review_to_wordlist(pair.hyp, True) for (pair, label) in train]
    train_label = np.array([[label] for (pair, label) in train])

    trainDataVecs = getCosFeatureVecs(train_text, train_hyp, model, num_features)

    i = np.isnan(trainDataVecs)
    trainDataVecs[i] = 0

    print("Creating feature vecs for test reviews")

    test_text = [KaggleWord2VecUtility.review_to_wordlist(pair.text, True) for (pair, label) in test]
    test_hyp = [KaggleWord2VecUtility.review_to_wordlist(pair.hyp, True) for (pair, label) in test]
    test_label = np.array([[label] for (pair, label) in test])

    testDataVecs = getCosFeatureVecs(test_text, train_hyp, model, num_features)

    i = np.isnan(testDataVecs)
    testDataVecs[i] = 0

    if lstm:
        (m, n) = trainDataVecs.shape
        trainDataVecs = np.reshape(trainDataVecs, (m, 1, n))
        (m, n) = testDataVecs.shape
        testDataVecs = np.reshape(testDataVecs, (m, 1, n))

        train_label = keras.utils.to_categorical(train_label)
        test_label = keras.utils.to_categorical(test_label)

    return (trainDataVecs, train_label, testDataVecs, test_label)

def rte_maxCosVecs(train, test, num_features, model, lstm=False):
    print("Creating feature vecs for training reviews")

    train_text = [KaggleWord2VecUtility.review_to_wordlist(pair.text, True) for (pair, label) in train]
    train_hyp = [KaggleWord2VecUtility.review_to_wordlist(pair.hyp, True) for (pair, label) in train]
    train_label = np.array([[label] for (pair, label) in train])

    trainDataVecs = getMaxCosFeatureVecs(train_text, train_hyp, model, num_features)
    # for i in trainDataVecs:
    #     print(i.shape)

    i = np.isnan(trainDataVecs)
    trainDataVecs[i] = 0

    print("Creating feature vecs for test reviews")

    test_text = [KaggleWord2VecUtility.review_to_wordlist(pair.text, True) for (pair, label) in test]
    test_hyp = [KaggleWord2VecUtility.review_to_wordlist(pair.hyp, True) for (pair, label) in test]
    test_label = np.array([[label] for (pair, label) in test])

    testDataVecs = getMaxCosFeatureVecs(test_text, train_hyp, model, num_features)
    # for i in testDataVecs:
    #     print(i.shape)

    i = np.isnan(testDataVecs)
    testDataVecs[i] = 0

    train_label = keras.utils.to_categorical(train_label)
    test_label = keras.utils.to_categorical(test_label)

    return (trainDataVecs, train_label, testDataVecs, test_label)

def rte_featureVecs(train, test, lstm=False):
    print("Creating feature vecs for training reviews")

    trainDataVecs = np.array([rte_classify.rte_features_vector(pair) for (pair, label) in train])
    train_label = np.array([[label] for (pair, label) in train])

    print("Creating feature vecs for test reviews")

    testDataVecs = np.array([rte_classify.rte_features_vector(pair) for (pair, label) in test])
    test_label = np.array([[label] for (pair, label) in test])

    if lstm:
        # trainDataVecs = np.array([[i] for i in trainDataVecs])
        # testDataVecs = np.array([[i] for i in testDataVecs])

        (m, n) = trainDataVecs.shape
        trainDataVecs = np.reshape(trainDataVecs, (m, 1, n))
        (m, n) = testDataVecs.shape
        testDataVecs = np.reshape(testDataVecs, (m, 1, n))

        train_label = keras.utils.to_categorical(train_label)
        test_label = keras.utils.to_categorical(test_label)

    return (trainDataVecs, train_label, testDataVecs, test_label)
