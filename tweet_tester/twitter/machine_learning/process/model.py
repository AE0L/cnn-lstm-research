from keras.layers import *
from keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from pre_processing.tokenizer import MAX_SEQUENCE_LENGTH
from pre_processing.word_vector import EMBEDDING_VECTOR_LENGTH
from tokenizer import CustomTokenizer
from word_vector import EMBEDDING_VECTOR_LENGTH, construct_embedding_matrix
from keras.initializers import Constant
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


tokenizer = CustomTokenizer()
tokenizer.train_tokenize()
tokenized_train = tokenizer.vectorize_input()
tokenized_val = tokenizer.vectorize_input()
tokenized_test = tokenizer.vectorize_input()

glove_file = ""
embedding_matrix = construct_embedding_matrix(
    glove_file, tokenizer.tokenizer.word_index)

# Embeddingg layer
embedding = Sequential()
model.add(Embedding(EMBEDDING_VECTOR_LENGTH, ))

# CNN model
cnn = Sequential()