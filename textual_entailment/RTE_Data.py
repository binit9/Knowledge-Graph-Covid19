import nltk
import pandas as pd
from sklearn import preprocessing


class Rte_Pair:
    def __init__(self, text, hyp):
        self.text = text
        self.hyp = hyp

def nltk_rte_data():
    """
    Classify RTEPairs
    """
    train = [(pair, pair.value) for pair in
             nltk.corpus.rte.pairs(['rte1_dev.xml', 'rte2_dev.xml',
                                    'rte3_dev.xml'])]
    test = [(pair, pair.value) for pair in
            nltk.corpus.rte.pairs(['rte1_test.xml', 'rte2_test.xml',
                                   'rte3_test.xml'])]

    return (train, test)

def snli_rte_data():
    # "C:/Users/shaun.c.dsouza/Documents/ai/text analysis/textanalysis_ai
    le = preprocessing.LabelEncoder()

    df_train = pd.read_csv('C:\\Users\\shaun.c.dsouza\\Documents\\ai\\text analysis\\snli_1.0\\snli_1.0_train.txt', sep='\t', header='infer')
    i = df_train.isna()
    df_train[i] = ""
    train = [(Rte_Pair(i[1], i[2]), int(i[0])) for i in zip(le.fit_transform(df_train.loc[:, 'entail']), df_train.loc[:, 'sentence1'], df_train.loc[:, 'sentence2'])]

    df_test = pd.read_csv('C:\\Users\\shaun.c.dsouza\\Documents\\ai\\text analysis\\snli_1.0\\snli_1.0_test.csv', sep=',', header='infer')
    i = df_test.isna()
    df_test[i] = ""
    test = [(Rte_Pair(i[1], i[2]), int(i[0])) for i in zip(df_test.loc[:, 'entail'], df_test.loc[:, 'sentence1'], df_test.loc[:, 'sentence2'])]

    return (train, test)

