from pyvi import ViTokenizer
import pandas as pd
from string import punctuation 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import operator
from sklearn.metrics import precision_recall_fscore_support 
import os.path
import matplotlib.pyplot as plt


class ProcessData:
    def __init__(self):
        self.stopwords = [line.rstrip('\n') for line in open('stopwords.txt')]
        for i in range(len(self.stopwords)):
            self.stopwords[i] = ViTokenizer.tokenize(self.stopwords[i])        
    
    def clear_data(self, sentences):
        newSentences = sentences.lower()
        newSentences = ''.join([c for c in newSentences if c not in punctuation])
        return newSentences

    def remove_stop_words(self, sentence):
        list_sentence = sentence.split(' ')
        list_sentence_new= [word for word in list_sentence if word not in self.stopwords]
        sentence_removed_stopword = ' '.join(list_sentence_new)
        return sentence_removed_stopword

    def words_segmentation(self, sentence):
        new_sentence = self.clear_data(sentence)
        new_sentence = ViTokenizer.tokenize(new_sentence)
        return self.remove_stop_words(new_sentence)
    def create_vocabulary(self):
        print('create vocabulary')
    
    
    def result_process_data(self, file_data):
        data = pd.read_csv(file_data, names = ['sentence', 'sentiment'])
        for i in range(data.shape[0]):
            data['sentence'][i] = self.words_segmentation(data['sentence'][i])
        return data

class NaiveBayes:
    def __init__(self, file_data_train):
        self.ProcessData = ProcessData()
        self.data = self.ProcessData.result_process_data(file_data_train)
        all_sentence = self.data['sentence'].values
        vectorizer = CountVectorizer()
        fre_all_words = vectorizer.fit_transform(all_sentence)
        self.all_words = vectorizer.get_feature_names()
        print('naivebayes')
    def cal_class_distribution(self):
        label_sentiment = self.data.groupby('sentiment').count().to_dict()
        label_sentiment_dict = label_sentiment['sentence']
        sum_label = sum(label_sentiment_dict.values())
        P_pos = label_sentiment_dict['positive'] / sum_label
        P_neg = label_sentiment_dict['negative'] / sum_label
        P_neu = label_sentiment_dict['neutral'] / sum_label
        return P_pos, P_neg, P_neu

    def training(self):
        cal_P_W_C_Pos = self.cal_P_W_C(1, 'positive')
        cal_P_W_C_Neg = self.cal_P_W_C(1, 'negative')
        cal_P_W_C_Neu = self.cal_P_W_C(1, 'neutral')

    def predict(self, sentence):
        sentence_input = self.ProcessData.words_segmentation(sentence)
        sentence_input = sentence_input.split(" ")
        P_pos, P_neg, P_neu = self.cal_class_distribution()
        
        if not os.path.exists("positive_model.csv"):
            self.training()
        if not os.path.exists("negative_model.csv"):
            self.training()
        if not os.path.exists("neutral_model.csv"):
            self.training()
        
        cal_P_W_C_Pos = pd.read_csv("positive_model.csv")
        cal_P_W_C_Neg = pd.read_csv("negative_model.csv")
        cal_P_W_C_Neu = pd.read_csv("neutral_model.csv")

        P_S_Pos = P_S_Neg = P_S_Neu = 1
        for w in sentence_input:
            if w in cal_P_W_C_Pos.columns:
                P_S_Pos = P_S_Pos * cal_P_W_C_Pos[w][0]
                P_S_Neg = P_S_Neg * cal_P_W_C_Neg[w][0]
                P_S_Neu = P_S_Neu * cal_P_W_C_Neu[w][0]

        P_C_Pos = P_pos * P_S_Pos
        P_C_Neg = P_neg * P_S_Neg
        P_C_Neu = P_neu * P_S_Neu
        P_all = {'positive':P_C_Pos, 'negative':P_C_Neg, 'neutral':P_C_Neu}
        result_predict = max(P_all.items(), key=operator.itemgetter(1))[0]
        return P_all, result_predict

    def cal_P_W_C(self, smooth, class_sentiment):
        df_class = self.data[self.data['sentiment'] == class_sentiment]
     
        all_sentences_of_class = df_class['sentence'].values
        class_vectorizer = CountVectorizer()
        fre_words_class = class_vectorizer.fit_transform(all_sentences_of_class)
        
        fre_class_df = pd.DataFrame(data=fre_words_class.toarray(), columns=class_vectorizer.get_feature_names())
        words_not_in_class = list(set(self.all_words) - set(class_vectorizer.get_feature_names()))
        len_class_df = all_sentences_of_class.shape[0]
        for i in words_not_in_class:
            fre_class_df[i] = pd.Series(np.zeros(len_class_df,dtype=int))

        sum_fre_words_class = fre_class_df.values.sum()
        list_P_W_C = [[]]
        for w in self.all_words:
            fre_w = fre_class_df[w].sum()
            P_word_class = (fre_w+smooth)/(sum_fre_words_class+len(self.all_words))
            list_P_W_C[0].append(P_word_class)
        P_sentence_class_df = pd.DataFrame(data = list_P_W_C,columns = self.all_words)
        path = class_sentiment+"_model.csv"
        P_sentence_class_df.to_csv(path)
        return P_sentence_class_df

    def test(self,file_test):
        print('test')
        data_test = self.ProcessData.result_process_data(file_test)
        result_class_test = data_test['sentiment'].values
        sentence_test = data_test['sentence'].values
        result_class_predict = []
        for s in sentence_test:
            P_all, predict = self.predict(s)
            result_class_predict.append(predict)
            print("{} ==> {}".format(s,predict))
        # print("result_class_test = {} \n result_class_predict = {}".format(result_class_test, result_class_predict))
        precision,recal,fscore,surport = precision_recall_fscore_support(result_class_test, result_class_predict, beta = 1,labels = ["positive","negative","neutral"])
        print("precision = {}, recal = {}, fscore = {}".format(precision,recal,fscore)) 
        self.draw(precision,recal,fscore)

    def draw(self, values_precision, values_recall, values_f1_score):
        n_groups = 3

        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.2
        opacity = 0.8

        rects1 = plt.bar(index, values_precision, bar_width,
        alpha=opacity,
        color='b',
        label='precision')

        rects2 = plt.bar(index + bar_width, values_recall, bar_width,
        alpha=opacity,
        color='r',
        label='Recall')

        rects2 = plt.bar(index + bar_width+ bar_width, values_f1_score, bar_width,
        alpha=opacity,
        color='g',
        label='F1-Score')

        plt.xlabel('Class')
        plt.ylabel('values')
        plt.title('Đánh giá mô hình')
        plt.xticks(index + bar_width, ('positive', 'negative', 'neutral'))
        plt.legend()

        plt.tight_layout()
        plt.show()

# nbc = NaiveBayes('traindata.csv')
# s = 'chăn bẩn không mặc áo bị ngứa'
# P_all, result_predict = nbc.predict(s)
# test = nbc.test('testdata.csv')



