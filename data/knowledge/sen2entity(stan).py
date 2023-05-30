import spacy
from spacy import displacy
from stanfordcorenlp import StanfordCoreNLP
import json
import tqdm

nlp = StanfordCoreNLP(r'/home/silverbullet/Tool/stanford-corenlp-4.2.2')
sentence_path = '../MSR-VTT/metadata/msrvtt_sentence.txt'
entity_rel_path = '../MSR-VTT/metadata/entity_total.txt'
# doc = nlp("The 22-year-old recently won ATP Challenger tournament.")


f = open(sentence_path, 'r')
w = open(entity_rel_path, 'w')
lines = f.readlines()
cnt = 0
for line in tqdm.tqdm(lines):
    sentence = line
    output = nlp.annotate(sentence, properties={"annotators": "tokenize,lemma,ssplit,pos,depparse,natlog,openie",
                                                "outputFormat": "json",
                                                'openie.triple.strict': 'true',
                                                'openie.max_entailments_per_clause': '1'
                                                })
    data = json.loads(output)
    # result = data['sentences'][0]['openie']
    # print(result)
    # print(data['sentences'][0].keys())
    # print(data['sentences'][0]['openie'])
    # print(data['sentences'][0]['tokens'])

    for i in range(len(data['sentences'])):
        result = [data["sentences"][i]["openie"] for item in data]
        lemmas = [data["sentences"][i]["tokens"] for item in data]
        cnt += 1
        # result = [data["sentences"][i]["openie"] for item in data]
        for g in result:
            for rel in g:
                l_relation, l_object, l_subject = '', '', ''
                span = str(rel['subjectSpan']), str(rel['objectSpan']), str(rel['relationSpan'])
                l_subject1 = lemmas[i][rel['subjectSpan'][0]:rel['subjectSpan'][1]][0:rel['subjectSpan'][1] - rel['subjectSpan'][0]]
                for h in range(rel['subjectSpan'][1] - rel['subjectSpan'][0]):
                    l_subject = l_subject1[h]['lemma'] + ' ' + l_subject
                l_object1 = lemmas[i][rel['objectSpan'][0]:rel['objectSpan'][1]][0:rel['objectSpan'][1] - rel['objectSpan'][0]]
                for s in range(rel['objectSpan'][1] - rel['objectSpan'][0]):
                    l_object = l_object1[s]['lemma'] + ' ' + l_object
                l_relation1 = lemmas[i][rel['relationSpan'][0]:rel['relationSpan'][1]][0:rel['relationSpan'][1] - rel['relationSpan'][0]]
                for j in range(rel['relationSpan'][1] - rel['relationSpan'][0]):
                    l_relation = l_relation1[j]['lemma'] + ' ' + l_relation
                # l_relation = lemmas[i][rel['relationSpan'][0]:rel['relationSpan'][1]][0][
                #              0:rel['relationSpan'][1] - rel['relationSpan'][0]]['lemma']
                # relationSent1 = rel['subject'], rel['object'], rel['relation']
                relationSent = l_subject, '&', l_object, '&', l_relation
                print(relationSent)
                w.writelines(relationSent)
                w.writelines('\n')

                print(str(cnt))
                # print(relationSent1)
print('total number is ' + str(cnt) + '\n')
print('reslut write to ' + entity_rel_path)
w.close()
f.close()

# sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))  # 语法树
# print('Dependency Parsing:', nlp.dependency_parse(sentence))  # 依存句法
# nlp.close()  # Do not forget to close! The backend server will consume a lot memery

# lines = f.readlines()
# for line in lines:
#     # print(line)
#     # print(type(line))
#     doc = nlp(line)
# displacy.render(doc, style='dep', jupyter=False)  # need to run in Jupyter

# break
# for tok in doc:
#     print(tok.text + "---------------->" + tok.pos_ + ' ' + tok.dep_ + ' ' + tok.tag_)
#     if tok.pos_ == 'NOUN':
#         w.writelines(tok.text + ' ')
#     # if tok.dep_ == 'pobj' or 'dobj':
#     #     w.writelines(tok.text + ' ')
#     if tok.pos_ == 'VERB':
#         w.writelines(tok.lemma_ + ' ')
#     if tok.pos_ == 'SPACE':
#         w.write('\n')
# print(tok.text + '------------>' + tok.dep_)

# for tok in doc:
#     print(tok.text, "...", tok.dep_)
