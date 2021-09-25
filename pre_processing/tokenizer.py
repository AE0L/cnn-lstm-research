from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.text import Tokenizer

TOP_K = 2000
MAX_SEQUENCE_LENGTH = 50


class CustomTokenizer:
    def __init__(self, train_texts):
        self.train_texts = train_texts
        self.tokenizer = Tokenizer(num_words=TOP_K)

    def train_tokenize(self):
        max_length = len(max(self.train_texts, key=len))
        self.max_length = min(max_length, MAX_SEQUENCE_LENGTH)
        self.tokenizer.fit_on_texts(self.train_texts)

    def vectorize_input(self, tweets):
        tweets = self.tokenizer.texts_to_sequences(tweets)
        tweets = sequence.pad_sequences(
            tweets, maxlen=self.max_length, truncating='post', padding='post')

        return tweets
