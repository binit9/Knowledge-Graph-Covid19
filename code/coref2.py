#import urllib.request
#from bs4 import BeautifulSoup
import spacy
import neuralcoref
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

# html = urllib.request.urlopen('https://www.law.cornell.edu/supremecourt/text/418/683').read()
# soup = BeautifulSoup(html, 'html.parser')


text = 'Angela lives in Boston. She is quite happy in that city.'
doc = nlp(text)
resolved_text = doc._.coref_resolved
print(resolved_text)

# sentences = [sent.string.strip() for sent in nlp(resolved_text).sents]
# output = [sent for sent in sentences if 'president' in 
          # (' '.join([token.lemma_.lower() for token in nlp(sent)]))]
# print('Fact count:', len(output))
# for fact in range(len(output)):
    # print(str(fact+1)+'.', output[fact])