import json
import pandas as pd

# dataset = 'msvd'
dataset = 'msrvtt'


def msvd_load_captions(caption_fpath):
    df = pd.read_csv(caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[pd.notnull(df['Description'])]
    captions = df['Description'].values
    return captions


if dataset == 'msvd':
    cnt = 0
    train_id_path = '../MSVD/metadata/train.list'
    train_sentence_path = '../MSVD/metadata/train.csv'
    sentence_path = '../MSVD/metadata/msvd_sentence.txt'
    w = open(sentence_path, 'a')
    sentence = msvd_load_captions(train_sentence_path)
    print(sentence.size)
    for i in range(sentence.size):
        w.writelines(sentence[i])
        w.write('\n')
        cnt += 1
    w.close()
    print(cnt)

if dataset == 'msrvtt':
    file_path = '../MSR-VTT/metadata/train.json'
    sentence_path = '../MSR-VTT/metadata/msrvtt_sentence.txt'
    list1 = []
    cnt = 0

    with open(file_path, 'r') as f:
        load_data = json.load(f)
        f.close()
    w = open(sentence_path, 'a')
    # print(load_data, type(load_data))
    for i in load_data.keys():
        list1.append(load_data[i])
    for j in range(len(list1)):
        for key in list1[j]:
            w.writelines(list1[j][key])
            w.write('\n')
            cnt += 1
            print(list1[j][key])
    w.close()
    print(cnt)

    # print(list1[99], len(list1),type(list1[99]))
    # print(list1, type(list1))
