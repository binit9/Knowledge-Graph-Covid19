
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D

np.random.seed(1)


# GRADED FUNCTION: pretrained_embedding_layer

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def Emojify_V2_th(input_shape, word_to_vec_map, word_to_index):
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
    sentence_indices_t = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer_t = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_t = embedding_layer_t(sentence_indices_t)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X_t = LSTM(128, return_sequences=True)(embeddings_t)
    # Add dropout with a probability of 0.5
    X_t = Dropout(0.5)(X_t)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X_t = LSTM(128)(X_t)
    # Add dropout with a probability of 0.5
    X_t = Dropout(0.5)(X_t)

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices_h = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer_h = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_h = embedding_layer_t(sentence_indices_h)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X_h = LSTM(128, return_sequences=True)(embeddings_h)
    # Add dropout with a probability of 0.5
    X_h = Dropout(0.5)(X_h)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X_h = LSTM(128)(X_h)
    # Add dropout with a probability of 0.5
    X_h = Dropout(0.5)(X_h)

    X = concatenate([X_t, X_h])

    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[sentence_indices_t, sentence_indices_h], outputs=X)

    ### END CODE HERE ###

    return model


def Emojify_V2_concat(input_shape, word_to_vec_map, word_to_index):
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
    sentence_indices_t = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer_t = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_t = embedding_layer_t(sentence_indices_t)

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices_h = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer_h = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_h = embedding_layer_t(sentence_indices_h)

    embeddings = concatenate([embeddings_t, embeddings_h])
    print(embeddings_t.shape, embeddings_h.shape, embeddings)

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

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[sentence_indices_t, sentence_indices_h], outputs=X)

    ### END CODE HERE ###

    return model


def Emojify_V2_cnn(input_shape, word_to_vec_map, word_to_index):
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
    sentence_indices_t = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer_t = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_t = embedding_layer_t(sentence_indices_t)

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices_h = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (â‰ˆ1 line)
    embedding_layer_h = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_h = embedding_layer_t(sentence_indices_h)

    embeddings = concatenate([embeddings_t, embeddings_h])
    print(embeddings_t.shape, embeddings_h.shape, embeddings)

    filters = 250
    kernel_size = 3
    hidden_dims = 250

    # X = Dropout(0.2)(embeddings)
    # X = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(X)
    # X = GlobalMaxPooling1D()(X)
    # X = Dense(hidden_dims)(X)
    # X = Dropout(0.2)(X)
    # X = Activation('relu')(X)
    # X = Dense(2)(X)
    # X = Activation('sigmoid')(X)

    pool_size = 4
    lstm_output_size = 70

    X = Dropout(0.25)(embeddings)
    X = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(X)
    X = MaxPooling1D(pool_size=pool_size)(X)
    X = LSTM(lstm_output_size)(X)
    X = Dense(2)(X)
    X = Activation('sigmoid')(X)



    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[sentence_indices_t, sentence_indices_h], outputs=X)

    ### END CODE HERE ###

    return model
