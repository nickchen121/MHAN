
rel_induction = '../MSR-VTT/OPENKE_file/rel.txt'
ent_induction = '../MSR-VTT/OPENKE_file/ent.txt'
entity2id = '../MSR-VTT/OPENKE_file/msrvtt/entity2id.txt'
relation2id = '../MSR-VTT/OPENKE_file/msrvtt/relation2id.txt'

f1 = open(rel_induction, 'r')
f2 = open(ent_induction, 'r')
w1 = open(relation2id, 'w')
w2 = open(entity2id, 'w')

rel_induction_lines = f1.readlines()
ent_induction_lines = f2.readlines()

cnt = 0
for line in rel_induction_lines:
    step1 = line.split('|')
    # w1.writelines("{0:100}{1}".format(step1[0], cnt))  # write rel2id
    w1.writelines(step1[0].replace(' ', '') + '\t' + str(cnt))
    w1.writelines('\n')
    cnt += 1
w1.close()

cnt = 0
for line in ent_induction_lines:
    step1 = line.split('|')
    # w2.writelines("{0:100}{1}".format((step1[0]), cnt))  # write ent2id
    w2.writelines(step1[0].replace(' ', '') + '\t' + str(cnt))
    w2.writelines('\n')
    cnt += 1
w2.close()
f1.close()
f2.close()
