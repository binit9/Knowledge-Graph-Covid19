import nltk
import pandas as pd
import numpy as np

train = [(pair, pair.value) for pair in
            nltk.corpus.rte.pairs(['rte1_dev.xml', 'rte2_dev.xml',
                                    'rte3_dev.xml'])]

test = [(pair, pair.value) for pair in
            nltk.corpus.rte.pairs(['rte1_test.xml', 'rte2_test.xml',
                                   'rte3_test.xml'])]

data = []
for i in test:
    data.append((i[0].text, i[0].hyp, i[1]))

df = pd.DataFrame(np.array(data))
print(df)
df.to_csv('test.csv')













