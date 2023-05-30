from tqdm import trange
entity_rel_path = '../MSR-VTT/metadata/entity_total.txt'
# relation_total_path = '../MSR-VTT/OPENKE_file/relation_total.txt'
# openke_train_file = '../MSR-VTT/OPENKE_file/train.txt'
# openke_valid_file = '../MSR-VTT/OPENKE_file/valid.txt'
# openke_test_file = '../MSR-VTT/OPENKE_file/test.txt'
entity_id_file = '../MSR-VTT/OPENKE_file/entity2id.txt'
relation_id_file = '../MSR-VTT/OPENKE_file/relation2id.txt'

f = open(entity_rel_path, 'r')
w1 = open(entity_id_file, 'w')
w2 = open(relation_id_file, 'w')
# w3 = open(relation_total_path, 'w')

lines = f.readlines()
entity_id_list = []
relation_id_list = []
relation_total_list = []
cnt = 0
rel = 0
for line in lines:
    list1 = line.split('&')
    for i in range(len(list1)):
        list1 = list1
        # relation_total_list.append(list1[i].strip().replace('\n', ''))
        # w3.writelines(list1[i].replace('\n', ''))
        if i == 0 or i == 1:
            if list1[i].strip() not in entity_id_list:
                entity_id_list.append(list1[i].strip())
        elif i == 2:
            if list1[i].replace('\n', '').strip() not in relation_id_list:
                relation_id_list.append(list1[i].replace('\n', '').strip())
            # w3.writelines('\n')
        # print(list1[i])

f.close()
for lens in range(len(entity_id_list)):
    # w1.writelines("{0:100}{1}".format(entity_id_list[lens], str(lens)))
    w1.writelines(entity_id_list[lens] + '\t' + str(lens))
    w1.writelines('\n')
    cnt += 1
for lens in range(len(relation_id_list)):
    # w2.writelines("{0:100}{1}".format(relation_id_list[lens], str(lens)))
    w2.writelines(relation_id_list[lens] + '\t' + str(lens))
    w2.writelines('\n')
    rel += 1
w1.close()
w2.close()
print(str(cnt) + '  ' + str(rel))
