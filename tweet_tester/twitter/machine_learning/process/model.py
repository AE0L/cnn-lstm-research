from keras.layers import Embedding, LSTM, Dense, Dropout
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

model = Sequential()

embedding = Embedding(len(tokenizer.tokenizer.word_index)+1,
                      EMBEDDING_VECTOR_LENGTH,
                      embeddings_initializer=Constant(embedding_matrix),
                      input_length=MAX_SEQUENCE_LENGTH,
                      trainable=False)

model.add(embedding)
model.add(Dropout(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(clipvalue=0.5)

model.compile(optimzer=optimizer,
              loss='binary_crossentropy',
              metrics=['acc', 'f1_m', 'precision_m', 'recall_m'])

history = model.fit(tokenized_train, y_train
                    batch_size=32,
                    epochs=20,
                    validation_data=(tokenized_val, y_val),
                    verbose=2)

loss, accuracy, f1_score, precision, recall = model.evaluate(
    tokenized_val, y_val, verbose=0)
