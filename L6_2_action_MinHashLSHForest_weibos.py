# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:14:28 2020

@author: Admin
"""

from datasketch import MinHash, MinHashLSHForest
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba.posseg as pseg
import random


"""0. 读取文件"""
file_path = r'C:\Users\Admin\Desktop\L6\weibos.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    weibos = f.read()

sentences = re.split('[。！？]', weibos.replace('\n', ''))

if sentences[-1] == '':
    del sentences[-1]

"""1.0 句子分词处理"""
# 获取停用词
def get_stopwords():
    sw = set()
    stop_words_file = r'C:\Users\Admin\Desktop\L6\stopword.txt'
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        for i in f:
            sw.add(i.strip()) 
    return list(sw)
stopwords = get_stopwords()

# 清除无意义符号
def clear(content):
    meaningless_symbols = re.compile(r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~—!，。？、￥…（）：【】《》‘’“”\s]+")
    return meaningless_symbols.sub('', content)

# 句子分词并去除其中停用词
def split_content(content):
    temp = clear(content)
    temp = temp.replace('\u200b', '')
    temp = pseg.cut(temp)
    result = ' '.join([i.word for i in temp if i not in stopwords])
    return result

# 对weibos内容进行分词
documents = []
for content in sentences:
    temp = split_content(content)
    documents.append(temp)

"""2.0 MinHash处理"""
# 计算MinHash
def get_minhash(content):
    mh = MinHash()
    for d in content:
        mh.update(d.encode('utf8'))
    return mh

# 创建MinHash及LSH Forest对象
minhash_list = []
forest = MinHashLSHForest()
for i in range(len(documents)):
    temp = get_minhash(documents[i])
    minhash_list.append(temp)
    forest.add(i, temp)
forest.index()

"""3.0 寻找某句子的相似对象"""
random.seed(666)
n = random.randint(0, len(sentences))   # 随机生成目标句子的index
target = sentences[n]
print("目标句子：", target)

split_target = split_content(target.replace('\u200b', ''))   # 对目标句子进行分词
minhash_target = get_minhash(split_target)   # 目标句子的MinHash处理

sim_results = forest.query(minhash_target, 3)   # 查找目标句子的Top-3相似句子
for i in range(len(sim_results)):
    print("-"*30)
    print('相似句子索引：', sim_results[i])
    print('与目标句子的Jaccard相似度：', minhash_target.jaccard(minhash_list[sim_results[i]]))
    print('相似句子内容：', sentences[sim_results[i]])

