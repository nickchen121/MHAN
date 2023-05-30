import turtle

##全局变量##
pi = 3.14159
# 词频排列显示个数
count = 10
# 单词频率数组-作为y轴数据
data = []
# 单词数组-作为x轴数据
words = []

import random

# --------------------------------------读取 字符统计函数
# filename:文件名
# data:出现次数从高到低排序
# words:名字从高到低排序
def read(filename, data, words):
    txt1 = open(filename, "r")  # 打开文件操作
    ##数据预处理   进行统计，统计后的数据装在字典word_spss中
    word_spss = process_read(txt1)
    pairs = list(word_spss.items())

    items = [[x, y] for (y, x) in pairs]
    items.sort()
    # 输出count个数词频结果
    for i in range(len(items) - 1, len(items) - count - 1, -1):
        print(items[i][1] + "\t" + str(items[i][0]))
        data.append(items[i][0])
        words.append(items[i][1])


# --------------------------------------随机颜色
# 生成一个随机的码
def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


# --------------------------------------将英文标点替换标点为空格
# 输入line
# 输出line
def replaceMark(line):
    for ch in line:
        if ch in "~@#$%^&*()_-+=<>?/,.:;{}[]|\'\"":
            line = line.replace(ch, " ")
    return line


# --------------------------------------数据预处理
# 输入txt1
# 输出处理后的数据的频率
def process_read(txt1):
    word_spss = {}
    for line in txt1:
        line = line.lower()  # 换成小写
        line = replaceMark(line)  # 去除符号
        words = line.split()  # 该函数利用split来区分空格 从而达到提取数组的效果
        for word in words:
            if word in word_spss:
                word_spss[word] += 1
            else:
                word_spss[word] = 1
    return word_spss


# --------------------------------------画图函数
def DIY_draw(data, words):
    turtle.title("词频结果统计图")
    turtle.setup(1200, 500, 0, 0)
    t = turtle.Turtle()

    sum = 0
    for ii in range(9):
        sum += data[ii]

    # 输出结果
    print(sum)
    # 开始画图
    x0 = 150
    coefficient = 2 * x0 / sum * 1.2;  # 系数确认
    t0 = -1
    pp = []
    for i in range(9):
        pp.append(words[i] + "[" + str(data[i]) + "]")

    print(pp[1])
    for i in range(9):
        t.color(randomcolor())
        t.penup()
        t.goto(x0, -data[i] * coefficient)
        # t.write(data[i], False, align="center", font=("Arial", 18, "normal"))
        x0 = x0 - data[i] * coefficient - data[i + 1] * coefficient
        t.pendown()
        t.begin_fill()
        t.circle(data[i] * coefficient)
        t.end_fill()
        t0 = t0 * (-1)

    x0 = 150
    for i in range(9):
        t.color(randomcolor())
        t.penup()
        t.goto(x0, -data[i] * coefficient - t0 * data[i] * coefficient * 1.2)
        t.write(pp[i], False, align="center", font=("Arial", 18, "normal"))
        x0 = x0 - data[i] * coefficient - data[i + 1] * coefficient
        t.pendown()
        t0 = t0 * (-1)
    t.down()


# --------------------------------------读取函数
def main():
    # 读取用户输入的数据
    read("../MSR-VTT/metadata/entity_total.txt", data, words)  # 调用read()函数
    DIY_draw(data, words)
    breakpoint()


main()
