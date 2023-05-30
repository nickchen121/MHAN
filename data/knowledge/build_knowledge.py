from tqdm import tqdm

rel_induction = '../MSR-VTT/OPENKE_file/rel.txt'  # rules of nlp result to rel
ent_induction = '../MSR-VTT/OPENKE_file/ent.txt'  # rules of nlp result to entity
target = '../MSR-VTT/OPENKE_file/entity_total_try.txt'
entity2id = '../MSR-VTT/OPENKE_file/msrvtt/entity2id.txt'
relation2id = '../MSR-VTT/OPENKE_file/msrvtt/relation2id.txt'
total = '../MSR-VTT/OPENKE_file/msrvtt/total_word.txt'
total_id = '../MSR-VTT/OPENKE_file/msrvtt/total_id.txt'
# test = '../MSR-VTT/OPENKE_file/msrvtt/test.txt'
# valid = '../MSR-VTT/OPENKE_file/msrvtt/valid.txt'

f1 = open(rel_induction, 'r')
f2 = open(ent_induction, 'r')
t = open(target, 'r')
# w1 = open(relation2id, 'w')
# w2 = open(entity2id, 'w')
# r1 = open(relation2id, 'r')
# r2 = open(entity2id, 'r')
####################
w3 = open(total, 'w')

rel_induction_lines = f1.readlines()
ent_induction_lines = f2.readlines()
target_lines = t.readlines()
# ur1 = r1.readlines()
# ur2 = r2.readlines()

for target_line in tqdm(target_lines):
    list1 = target_line.split('&')
    for i in range(len(list1)):
        if i == 0 or i == 1:
            no_match = True
            for line in ent_induction_lines:
                if not no_match:  # avoid repeat match string
                    break
                step1 = line.split('|')

                step2 = step1[1].replace('\n', '').split('#')
                # if list1[i].strip() in step2:
                for j in step2:
                    if j in list1[i].strip() and j != '':
                        w3.write(step1[0].replace(' ', '') + ',')
                        no_match = False
                        break

                # print(step2)
            if no_match:
                w3.write('none,')
            # w1.close()
        elif i == 2:
            no_match = True
            for line in rel_induction_lines:
                if not no_match:
                    break
                step1 = line.split('|')
                step2 = step1[1].replace('\n', '').split('#')
                for j in step2:
                    if j in list1[i].strip():
                        w3.write(step1[0].replace(' ', '') + '\n')
                        no_match = False
                        break
            if no_match:
                w3.write('none\n')

            # w2.close()
w3.close()
############
# tran total.txt to id
r1 = open(relation2id, 'r')
r2 = open(entity2id, 'r')
r3 = open(total, 'r')
total_id = open(total_id, 'w')
# w_test = open(test, 'w')
# w_valid = open(valid, 'w')

relid = r1.readlines()
entid = r2.readlines()
reldict = {}
entdict = {}
for line in relid:
    rel_id = line.split('\t')
    # print(rel_id)
    reldict[rel_id[0]] = int(rel_id[-1].replace('\n', ''))
for line in entid:
    ent_id = line.split('\t')
    entdict[ent_id[0]] = int(ent_id[-1].replace('\n', ''))
print(reldict, '\n', entdict)
r1.close()
r2.close()
total = r3.readlines()
for line in tqdm(total):
    word = line.split(',')
    word[2] = word[2].replace('\n', '')
    if word[0] == 'none' or word[1] == 'none' or word[2] == 'none':
        continue
    for i in range(3):
        if i == 0 or i == 1:
            assert 0 <= entdict[word[i]] <= 80, "entity id get some trouble!!!"
            total_id.write(str(entdict[word[i]]) + ' ')
        elif i == 2:
            assert 0 <= reldict[word[i]] <= 20, "relation id get some trouble!!!"
            total_id.write(str(reldict[word[i]]) + '\n')
