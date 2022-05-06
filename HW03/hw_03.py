import jieba
import os
import re
import time
import math
import numpy as np
import random
from gensim import corpora, models
from collections import defaultdict
from sklearn.svm import SVC


def data_preprocessing(data_roots, abandon_stop_words):
    listdir = os.listdir(data_roots)

    char_to_be_replaced = "\n `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
                          "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
                          "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"
    char_to_be_replaced = list(char_to_be_replaced)

    txt_corpus = []
    label_idxes = []
    label_words = []

    label_idx = 0
    label_idx_to_words = dict()

    stop_words_list = []
    for tmp_file_name in os.listdir("/codes/NLP_homework/NLP_homework3/stopwords/"):      # replace this path with the stopwords path
        with open("/codes/NLP_homework/NLP_homework3/stopwords/"+tmp_file_name, "r", encoding="utf-8", errors="ignore") as f:
            stop_words_list.extend([word.strip('\n') for word in f.readlines()])

    for tmp_file_name in listdir:
        if tmp_file_name == "inf.txt":
            continue
        path = os.path.join(data_roots, tmp_file_name)
        if os.path.isfile(path):
            with open(path, "r", encoding="gbk", errors="ignore") as tmp_file:
                tmp_file_context = tmp_file.read()
                for tmp_char in char_to_be_replaced:
                    tmp_file_context = tmp_file_context.replace(tmp_char, "")
                tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")
                if abandon_stop_words:
                    for tmp_char in stop_words_list:
                        tmp_file_context = tmp_file_context.replace(tmp_char, "")
                txt_corpus.append(tmp_file_context)
                label_idxes.append(label_idx)
                label_words.append(tmp_file_name.split(".txt")[0])
                label_idx_to_words[label_idx] = tmp_file_name.split(".txt")[0]
                label_idx += 1

    return txt_corpus, label_idxes, label_words, label_idx_to_words


if __name__ == '__main__':
    num_topics = 50
    num_docs = 200
    len_per_doc = 500
    abandon_stop_words = True
    print("主题数：{}，段落(文档)数：{}，每段话字数：{}，是否去除停用词：{}".format(num_topics, num_docs, len_per_doc, "yes" if abandon_stop_words else "no"))
    print("Preparing data...")
    data_roots = '/codes/NLP_homework/NLP_homework1/txt_files/'  # replace this path with the txt files path
    txt_corpus, label_idxes, label_words, label_idx_to_words = data_preprocessing(data_roots, abandon_stop_words)
    whole_samples = []

    #### get training samples and testing samples
    for i in range(len(txt_corpus)):
        for j in range(num_docs//len(txt_corpus) + 1):
            tmp_start = random.randint(0, len(txt_corpus[i])-len_per_doc-1)
            tmp_sample = list(jieba.cut(txt_corpus[i][tmp_start:tmp_start + len_per_doc]))
            whole_samples.append((label_idxes[i], tmp_sample))


    random.shuffle(whole_samples)
    whole_samples = whole_samples[:num_docs]
    train_data, train_label = [], []
    test_data, test_label = [], []

    for i in range(int(len(whole_samples) * (1 - 0.2))):
        train_data.append(whole_samples[i][1])
        train_label.append(whole_samples[i][0])
    for i in range(int(len(whole_samples) * (1 - 0.2)), len(whole_samples)):
        test_data.append(whole_samples[i][1])
        test_label.append(whole_samples[i][0])


    #### train lda
    dictionary = corpora.Dictionary(train_data)
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in train_data]
    print("Trainng LDA model...")
    lda = models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics=num_topics)


    #### train svm classifier for correct label
    train_topic_distribution = lda.get_document_topics(lda_corpus_train)
    train_features = np.zeros((len(train_data), num_topics))
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            train_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]


    print("Training SVM classifier...")
    assert len(train_label) == len(train_features)
    train_label = np.array(train_label)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(train_features, train_label)
    print("Prediction accuracy of training samples is {:.4f}.".format(sum(classifier.predict(train_features) == train_label) / len(train_label)))


    #### testing
    lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in test_data]
    test_topic_distribution = lda.get_document_topics(lda_corpus_test)
    test_features = np.zeros((len(test_data), num_topics))
    for i in range(len(test_topic_distribution)):
        tmp_topic_distribution = test_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            test_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]
    assert len(test_label) == len(test_features)
    test_label = np.array(test_label)
    print("Prediction accuracy of testing samples is {:.4f}.".format(sum(classifier.predict(test_features) == test_label) / len(test_label)))