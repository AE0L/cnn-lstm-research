from nltk.corpus import stopwords
import string
from spellchecker import SpellChecker
import re

spell = SpellChecker()
stop_words = stopwords.words('english')


def get_text_file(filename):
    """Return specified file into a string"""
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text


def correct_spellings(x, spell=spell):
    """Correct the mispelled words in a string"""
    x = x.split()
    mispelled = spell.unknown(x)
    result = map(lambda word: spell.correction(
        word) if word in mispelled else word, x)
    return " ".join(result)


def clean_tweet(tweet, correct_spelling=True, remove_emojis=True, remove_stop_words=True):
    """Clean a tweet"""
    # lower and strip leading and trailing whitespace
    tweet = tweet.lower().strip()

    # remove URLs
    URL = re.compile(r'https?://\S+|www\.\S+')
    tweet = URL.sub(r'', tweet)

    # remove HTML
    HTML = re.compile(r'<.*?>')
    tweet = HTML.sub(r'', tweet)

    # remove punctuation
    puncts = str.maketrans('', '', string.punctuation)
    tweet = tweet.translate(puncts)

    if correct_spelling:
        tweet = correct_spellings(tweet)

    if remove_emojis:
        tweet = tweet.encode('ascii', 'ignore').decode('UTF-8').strip()

    if remove_stop_words:
        tweet = ' '.join([word for word in tweet.split()
                         if word not in stop_words])

    return tweet
