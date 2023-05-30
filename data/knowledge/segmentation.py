import os
import random


def ReadFileDatas(original_filename):
    file = open(original_filename, 'r+')
    FileNameList = file.readlines()
    random.shuffle(FileNameList)
    file.close()
    print("Nmber of dataset:", len(FileNameList))
    return FileNameList


def TrainValTestFile(FileNameList):
    i = 0
    j = len(FileNameList)
    l_train, l_val, l_test = [], [], []
    for line in FileNameList:
        if i < (j * 0.6):
            i += 1
            l_train.append(line)
        elif i < (j * 0.8):
            i += 1
            l_val.append(line)
        else:
            i += 1
            l_test.append(line)
    print("total number:%d, now has create train,val,test dataset" % i)
    return l_train, l_val, l_test


def WriteDatasToFile(listInfo, new_filename):
    file_handle = open(new_filename, 'w')
    for str_Result in listInfo:
        file_handle.write(str_Result)
    file_handle.close()
    print('write %s file successes!' % new_filename)


if __name__ == "__main__":
    # listFileInfo = ReadFileDatas('../MSR-VTT/OPENKE_file/msrvtt/b.txt')
    listFileInfo = ReadFileDatas('../MSR-VTT/OPENKE_file/msrvtt/total_id.txt')
    l_train, l_val, l_test = TrainValTestFile(listFileInfo)
    WriteDatasToFile(l_train, '../MSR-VTT/OPENKE_file/msrvtt/train2id.txt')
    WriteDatasToFile(l_val, '../MSR-VTT/OPENKE_file/msrvtt/valid2id.txt')
    WriteDatasToFile(l_test, '../MSR-VTT/OPENKE_file/msrvtt/test2id.txt')
