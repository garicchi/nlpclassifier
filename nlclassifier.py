import os
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np



def tokenize(text):
    """ MeCab で分かち書きした結果をトークンとして返す """
    wakati = MeCab.Tagger("-O wakati")
    return wakati.parse(text)

token_dict = []
# ひとつひとつファイルを読み込んで
# ファイル名に対して語彙群のディクショナリを生成する
token_dict.append('あらゆる現実をすべて自分の方へねじ曲げたのだ')
token_dict.append('一週間ばかり、ニューヨークを取材した。')
token_dict.append('テレビゲームやパソコンで、ねじ曲げたのだ')
token_dict.append('ロンドンを取材したい')
token_dict.append('ねじ曲げたいのであった')

label = ['true','false','true','false']


# scikit-learn の TF-IDF ベクタライザーを使う
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict)

print(token_dict)
print(tfs.toarray())

row,col = tfs.shape
pre = tfs[row-1]
tfs = tfs[:row-1]

C = 1.
kernel = 'rbf'
gamma  = 0.01

classifier = SVC(C=C, kernel=kernel, gamma=gamma)
classifier.fit(tfs, label)
pred_y = classifier.predict(pre)
print(pred_y);