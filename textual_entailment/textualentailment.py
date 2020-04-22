import nltk
import logging
from gensim.models import Word2Vec
import sys
import os
sys.path.append(os.path.abspath('.'))
# print(sys.path)
from text_analysis.textual_entailment import rte_classify
from text_analysis.textual_entailment import RTE_Data
from text_analysis.textual_entailment import Word2Vec_AverageVectors
from text_analysis.textual_entailment import Word2Vec_Vectors
from text_analysis.textual_entailment import Train_Vectors
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
from text_analysis.textual_entailment.KaggleWord2VecUtility import KaggleWord2VecUtility
from keras.models import load_model
import json
import keras
import traceback

class TextualEntailment():
    lstm = True

    def word2vec_model(self):
        min_word_count = 40  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model (this will take some time)
        print("Training Word2Vec model...")
        model_name = "text_analysis/Word_Embeddings/brown_model"
        # model.save(model_name)
        model = Word2Vec.load(model_name)

        return model

    def rte_classifier_w2v(self):
        (train, test) = RTE_Data.nltk_rte_data()

        # ****** Set parameters and train the word2vec model
        #
        # Import the built-in logging module and configure it so that Word2Vec
        # creates nice output messages
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                            level=logging.INFO)

        # Set values for various parameters
        num_features = 100  # Word vector dimensionality
        model_w2v = self.word2vec_model()

        (trainDataVecs, train_label, testDataVecs, test_label) = Word2Vec_Vectors.rte_avgVecs(train, test, num_features, model_w2v, self.lstm)
        # (trainDataVecs, train_label, testDataVecs, test_label) = Word2Vec_Vectors.rte_cosVecs(train, test, num_features, model_w2v, lstm)
        # (trainDataVecs, train_label, testDataVecs, test_label) = Word2Vec_Vectors.rte_featureVecs(train, test, lstm)
        # (trainDataVecs, train_label, testDataVecs, test_label) = Word2Vec_Vectors.rte_maxCosVecs(train, test, num_features, model_w2v, lstm)

        print(trainDataVecs.shape)
        print(train_label.shape)
        print(testDataVecs.shape)
        print(test_label.shape)

        model_name = 'text_analysis/textual_entailment/AverageVectors.h5'

        if True:
            if self.lstm:
                Train_Vectors.trainLSTM(model_name, trainDataVecs, train_label)
            else:
                Train_Vectors.trainSequential(model_name, trainDataVecs, train_label)
                # Train_Vectors.trainMLP(model_name, trainDataVecs, train_label)
                # Train_Vectors.trainCNN(model_name, trainDataVecs, train_label)

        model = load_model(model_name)

        # print(trainDataVecs.shape)
        # print("train_label", train_label[:10])

        # predict_prob = model.predict(trainDataVecs[:10])
        # idx = np.argmax(predict_prob, axis=1)
        # proba = np.amax(predict_prob, axis=1)
        # result = keras.utils.to_categorical(idx, num_classes=2)

        # predict_prob = model.predict(testDataVecs)
        # idx = np.argmax(predict_prob, axis=1)
        # proba = np.amax(predict_prob, axis=1)
        # result = keras.utils.to_categorical(idx, num_classes=2)
        #
        print("test_label", test_label[:10])
        # print("predict_prob", predict_prob[:10])
        # print("argmax", idx[:10])
        # print("proba", proba[:10])

        if self.lstm:
            result = keras.utils.to_categorical(model.predict_classes(testDataVecs), num_classes=2)
        else:
            result = model.predict_classes(testDataVecs)
        print("result", result[:10])

        loss, acc = model.evaluate(testDataVecs, test_label)
        # print(model.metrics_names)
        print('Test loss:', loss)
        print('Test acc:', acc)

        print(metrics.accuracy_score(test_label, result))
        print(metrics.classification_report(test_label, result))


    def rte_classify_w2v(self, text, hyp):
        # pair = nltk.corpus.rte.pairs(['rte1_test.xml'])[22]

        # print(pair.value)
        # print(pair.text)
        # print(pair.hyp)

        num_features = 100  # Word vector dimensionality
        model_w2v = self.word2vec_model()

        # input_text = [KaggleWord2VecUtility.review_to_wordlist(text, True)]
        # input_hyp = [KaggleWord2VecUtility.review_to_wordlist(hyp, True)]

        # print(input_text)
        # print(input_hyp)

        # inputTextDataVecs = Word2Vec_AverageVectors.getAvgFeatureVecs(input_text, model_w2v, num_features, self.lstm)
        # inputHypDataVecs = Word2Vec_AverageVectors.getAvgFeatureVecs(input_hyp, model_w2v, num_features, self.lstm)
        # inputDataVecs = np.concatenate((inputTextDataVecs, inputHypDataVecs), axis=1)

        # i = np.isnan(inputDataVecs)
        # inputDataVecs[i] = 0

        test = [(RTE_Data.Rte_Pair(text, hyp), 1)]
        (trainDataVecs, train_label, testDataVecs, test_label) = Word2Vec_Vectors.rte_avgVecs(test, test, num_features, model_w2v, self.lstm)

        model_name = 'text_analysis/textual_entailment/AverageVectors.h5'
        model = load_model(model_name)

        print(testDataVecs.shape)
        # if self.lstm:
        #     a = model.predict_classes(testDataVecs)
        #     print("test",a.shape,a)
        #     result = keras.utils.to_categorical(model.predict_classes(testDataVecs), num_classes=2)
        # else:
        # result = model.predict_classes(testDataVecs)
        # print("result",result.shape,result)
        # print(model.predict_proba(testDataVecs), model.predict_proba(testDataVecs)[0][result[0]])

        predict_prob = model.predict(testDataVecs)
        print("predict_prob", predict_prob)

        idx = np.argmax(predict_prob, axis=1)
        proba = np.amax(predict_prob, axis=1)
        print(idx, proba)

        return {"text": text, "hyp": hyp, "entail": str(idx[0]), "proba": str(proba[0])}


if __name__ == '__main__':
    try:
        entail = TextualEntailment()
        entail.rte_classifier_w2v()
    except:
        traceback.print_exc()

    # entail.rte_classify_w2v()

    # RTE_Data.snli_rte_data()

