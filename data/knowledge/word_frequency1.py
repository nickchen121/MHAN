"""
本脚本实现，合并几个英文文本，并且统计词频。
脚本定义了几个函数：
1、文件读取函数readFile（读取文件，输出每个文件的词频）；
2、元素为词频的字典的合并函数，并且实现相同词的词频相加，返回全部词频；
3、调试部分，利用了高阶函数：map,reduce;
4、最后实现格式化输出，输入结果如图片所示。
"""
import functools


# 定义文件读取函数，并且输出元素为词频的字典
def readFile(file_name):
    y = []
    with open(file_name, 'r', encoding="utf-8") as f:
        x = f.readlines()
    for line in x:
        # line.replace('&', '')
        y.extend(line.split())
    word_list2 = []

    # 单词格式化：去掉分词之后部分英文前后附带的标点符号
    for word in y:
        # last character of each word
        word1 = word

        # use a list of punctuation marks
        while True:
            lastchar = word1[-1:]
            if lastchar in [",", ".", "!", "?", ";", '"']:
                word2 = word1.rstrip(lastchar)
                word1 = word2
            else:
                word2 = word1
                break

        while True:
            firstchar = word2[0]
            if firstchar in [",", ".", "!", "?", ";", '"']:
                word3 = word2.lstrip(firstchar)
                word2 = word3
            else:
                word3 = word2
                break
                # build a wordList of lower case modified words
        word_list2.append(word3)
    # 统计词频
    tf = {}
    for word in word_list2:
        word = word.lower().replace('#', ' ').replace('&', '')
        # print(word)
        word = ''.join(word.split())
        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1
    return tf


def get_counts(words):
    tf = {}
    for word in words:
        word = word.lower()
        # print(word)
        word = ''.join(word.split())
        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1


# 合并两个字典的方法1
def merge1(dic1, dic2):
    for k, v in dic1.items():
        if k in dic2.keys():
            dic2[k] += v
        else:
            dic2[k] = v
    # print(dic2)
    return dic2


# 合并两个字典的方法2
def merge2(dic1, dic2):
    from collections import Counter
    counts = Counter(dic1) + Counter(dic2)
    return counts


# 获得前n个最热词和词频
def top_counts(word_list, n=100):
    value_key_pairs = sorted([(count, tz) for tz, count in word_list.items()], reverse=True)
    return value_key_pairs[:n]
    # print(value_key_pairs[:n])


# 测试部分
if __name__ == '__main__':
    # file_list = ['../MSR-VTT/metadata/entity_total.txt']
    file_list = ['../MSR-VTT/OPENKE_file/relation2id.txt']
    word_frequency_file = '../MSR-VTT/OPENKE_file/rel_frequency.txt'
    w = open(word_frequency_file, 'w')
    cc = map(readFile, file_list)
    word_list = functools.reduce(merge2, cc)
    top_counts = top_counts(word_list, 200)
    # print(top_counts)
    print("最常用的单词排行榜:")
    for word in top_counts[0:200]:
        w.writelines("{0:100}{1}".format(word[1], word[0]))
        w.writelines('\n')
        print("{0:100}{1}".format(word[1], word[0]))
