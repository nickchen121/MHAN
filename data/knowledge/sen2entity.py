import spacy
from spacy import displacy
from stanfordcorenlp import StanfordCoreNLP

nlp = spacy.load('en_core_web_sm')
# nlp1 = StanfordCoreNLP(r'/home/silverbullet/Tool/stanford-corenlp-4.2.2')
sentence_path = '../MSR-VTT/metadata/sentence.txt'
entity_path = '../MSR-VTT/metadata/entity_total.txt'
# doc = nlp("The 22-year-old recently won ATP Challenger tournament.")

f = open(sentence_path, 'r')
w = open(entity_path, 'w')

lines = f.readlines()
for line in lines:
    # print(line)
    # print(type(line))
    doc = nlp(line)
    # displacy.render(doc, style='dep', jupyter=False)  # need to run in Jupyter

    # break
    for tok in doc:
        print(tok.text + "---------------->" + tok.pos_ + ' ' + tok.dep_ + ' ' + tok.tag_)
        if tok.pos_ == 'NOUN':
            w.writelines(tok.text + ' ')
        # if tok.dep_ == 'pobj' or 'dobj':
        #     w.writelines(tok.text + ' ')
        if tok.pos_ == 'VERB':
            w.writelines(tok.lemma_ + ' ')
        if tok.pos_ == 'SPACE':
            w.write('\n')
        # print(tok.text + '------------>' + tok.dep_)

# for tok in doc:
#     print(tok.text, "...", tok.dep_)
