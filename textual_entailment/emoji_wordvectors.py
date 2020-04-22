
import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt


X_train_h, X_train_t, Y_train_e = read_csv_ht('text_analysis/textual_entailment/data/train.csv', rte=True)
X_test_h, X_test_t, Y_test_e = read_csv_ht('text_analysis/textual_entailment/data/test.csv', rte=True)


# In[5]:

maxLen = len(max(X_train_h, key=len).split())
print(maxLen)

index = 1
print(X_train_h[index], label_to_emoji(Y_train_e[index]))



Y_oh_train_e = convert_to_one_hot(Y_train_e, C = 2)
Y_oh_test_e = convert_to_one_hot(Y_test_e, C = 2)

index = 50
print(Y_train_e[index], "is converted into one hot", Y_oh_train_e[index])

# word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('text_analysis/textual_entailment/data/glove.6B.50d.txt')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('text_analysis/Word_Embeddings/glove_100d.txt')

word = "cucumber"
index = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(index) + "th word in the vocabulary is", index_to_word[index])





import numpy as np
np.random.seed(0)
import re
import sys
import os
sys.path.append(os.path.abspath('.'))
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from text_analysis.textual_entailment import Train_Vectors_Glove
np.random.seed(1)


# GRADED FUNCTION: sentences_to_indices

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (â‰ˆ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        review_text = re.sub("[^a-zA-Z]", " ", X[i])

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = review_text.lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[w]
            except:
                # X_indices[i, j] = 0
                print("err ", w)
            # print(type(word_to_index[w]))
            # print(index_to_word[0])
            # Increment j to j + 1
            j = j + 1

    ### END CODE HERE ###

    return X_indices


X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)




# In[26]:

# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


# GRADED FUNCTION: Emojify_V2

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    print("embeddings", embeddings.shape)
    print("sentence_indices", sentence_indices.shape)
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    print("X", X.shape)
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    ### END CODE HERE ###

    return model


model = Train_Vectors_Glove.Emojify_V2_cnn((maxLen,), word_to_vec_map, word_to_index)
model.summary()


# As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use. Compile your model using `categorical_crossentropy` loss, `adam` optimizer and `['accuracy']` metrics:

# In[34]:

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# It's time to train your model. Your Emojifier-V2 `model` takes as input an array of shape (`m`, `max_len`) and outputs probability vectors of shape (`m`, `number of classes`). We thus have to convert X_train (array of sentences as strings) to X_train_indices (array of sentences as list of word indices), and Y_train (labels as indices) to Y_train_oh (labels as one-hot vectors).

# In[35]:

X_train_h_indices = sentences_to_indices(X_train_h, word_to_index, maxLen)
X_train_t_indices = sentences_to_indices(X_train_t, word_to_index, maxLen)
Y_train_e_oh = convert_to_one_hot(Y_train_e, C = 2)


# Fit the Keras model on `X_train_indices` and `Y_train_oh`. We will use `epochs = 50` and `batch_size = 32`.

# In[36]:

model.fit([X_train_h_indices, X_train_t_indices], Y_train_e_oh, epochs = 50, batch_size = 32, shuffle=True)


# Your model should perform close to **100% accuracy** on the training set. The exact accuracy you get may be a little different. Run the following cell to evaluate your model on the test set.

# In[37]:

X_test_h_indices = sentences_to_indices(X_test_h, word_to_index, max_len = maxLen)
X_test_t_indices = sentences_to_indices(X_test_t, word_to_index, max_len = maxLen)
Y_test_e_oh = convert_to_one_hot(Y_test_e, C = 2)
loss, acc = model.evaluate([X_test_h_indices, X_test_t_indices], Y_test_e_oh)
print()
print("Test accuracy = ", acc)
exit()

# You should get a test accuracy between 80% and 95%. Run the cell below to see the mislabelled examples.

# In[38]:

# This code allows you to see the mislabelled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())


# Now you can try it on your own example. Write your own sentence below.

# In[39]:

# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.
x_test = np.array(['not feeling happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))

