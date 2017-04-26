'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

import kq.dataset

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################
## set directories and parameters
########################################
EMBEDDING_FILE = '/Users/richardweiss/Datasets/glove.6B.300d.txt'
TRAIN_DATA_FILE = '/Users/richardweiss/Datasets/Kaggle-Quora/train.csv'
TEST_DATA_FILE = '/Users/richardweiss/Datasets/Kaggle-Quora/test.csv'
os.makedirs('cache/kagglekeras/', exist_ok=True)
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)



train_texts_1 = []
train_texts_2 = []
train_labels = []

train_dataset = kq.dataset.Dataset().load_named('train')
for _, row in tqdm(train_dataset.iterrows(), total=train_dataset.shape[0]):
    train_texts_1.append(text_to_wordlist(row.question1_raw))
    train_texts_2.append(text_to_wordlist(row.question2_raw))
    train_labels.append(row.is_duplicate)
print('Found %s texts in train.csv' % len(train_texts_1))


valid_texts_1 = [] 
valid_texts_2 = []
valid_labels = []

valid_dataset = kq.dataset.Dataset().load_named('valid')
for _, row in tqdm(valid_dataset.iterrows(), total=valid_dataset.shape[0]):
    valid_texts_1.append(text_to_wordlist(row.question1_raw))
    valid_texts_2.append(text_to_wordlist(row.question2_raw))
    valid_labels.append(row.is_duplicate)
print('Found %s texts in valid.csv' % len(valid_texts_1))

merge_texts_1 = [] 
merge_texts_2 = []
merge_labels = []

merge_dataset = kq.dataset.Dataset().load_named('merge')
for _, row in tqdm(merge_dataset.iterrows(), total=merge_dataset.shape[0]):
    merge_texts_1.append(text_to_wordlist(row.question1_raw))
    merge_texts_2.append(text_to_wordlist(row.question2_raw))
    merge_labels.append(row.is_duplicate)
print('Found %s texts in merge.csv' % len(merge_texts_1))

test_texts_1 = [] 
test_texts_2 = []

test_dataset = kq.dataset.Dataset().load_named('test')
for _, row in tqdm(test_dataset.iterrows(), total=test_dataset.shape[0]):
    test_texts_1.append(text_to_wordlist(row.question1_raw))
    test_texts_2.append(text_to_wordlist(row.question2_raw))
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_texts_1 + valid_texts_1 + merge_texts_1 + test_texts_1 +
                       train_texts_2 + valid_texts_2 + merge_texts_2 + test_texts_2)

train_sequences_1 = tokenizer.texts_to_sequences(train_texts_1)
train_sequences_2 = tokenizer.texts_to_sequences(train_texts_2)
valid_sequences_1 = tokenizer.texts_to_sequences(valid_texts_1)
valid_sequences_2 = tokenizer.texts_to_sequences(valid_texts_2)
merge_sequences_1 = tokenizer.texts_to_sequences(merge_texts_1)
merge_sequences_2 = tokenizer.texts_to_sequences(merge_texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

train_data_1 = pad_sequences(train_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
train_data_2 = pad_sequences(train_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
valid_data_1 = pad_sequences(valid_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
valid_data_2 = pad_sequences(valid_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
merge_data_1 = pad_sequences(merge_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
merge_data_2 = pad_sequences(merge_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
merge_labels = np.array(merge_labels)

test_ids = np.arange(len(test_sequences_1))

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

English = spacy.en.English()
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    lex = English.vocab[word]
    if not lex.is_oov:
        embedding_matrix[i] = lex.vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(train_data_1))

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
    valid_weights = np.asarray([class_weight[l] for l in valid_labels])
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = 'cache/kagglekeras/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([train_data_1, train_data_2], train_labels,
        validation_data=([valid_data_1, valid_data_2], valid_labels, valid_weights),
        epochs=200, batch_size=2048, shuffle=True,
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

#model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################

merge_preds = model.predict([merge_data_1, merge_data_2], batch_size=8192, verbose=1)
merge_preds += model.predict([merge_data_2, merge_data_1], batch_size=8192, verbose=1)
merge_preds /= 2

np.save('cache/kagglekeras/merge.npy', merge_preds)

preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('cache/kagglekeras/%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
