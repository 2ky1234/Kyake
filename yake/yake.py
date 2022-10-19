# -*- coding: utf-8 -*-

"""Main module."""

import string
import os
from jaccard_similarity import isReDuplicated
import jellyfish
from .Levenshtein import Levenshtein

#from .datarepresentation import DataCore
from .datarepresentation_korea import DataCore

class KeywordExtractor(object):

    def __init__(self, lan="ko", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, features=None, stopwords=None):
        self.lan = lan

        dir_path = os.path.dirname(os.path.realpath(__file__))

        local_path = os.path.join("StopwordsList", "stopwords_%s.txt" % lan[:2].lower())

        if os.path.exists(os.path.join(dir_path,local_path)) == False:
            local_path = os.path.join("StopwordsList", "stopwords_noLang.txt")
        
        resource_path = os.path.join(dir_path,local_path)

        if stopwords is None:
            try:
                with open(resource_path, encoding='utf-8') as stop_fil:
                    self.stopword_set = set( stop_fil.read().lower().split("\n") )
            except:
                print('Warning, read stopword list as ISO-8859-1')
                with open(resource_path, encoding='ISO-8859-1') as stop_fil:
                    self.stopword_set = set( stop_fil.read().lower().split("\n") )
        else:
            self.stopword_set = set(stopwords)

        self.n = n
        self.top = top
        self.dedupLim = dedupLim
        self.features = features
        self.windowsSize = windowsSize
        if dedupFunc == 'jaro_winkler' or dedupFunc == 'jaro':
            self.dedu_function = self.jaro
            print('self.dedu_function jaro 테스트 :', self.dedu_function)
        elif dedupFunc.lower() == 'sequencematcher' or dedupFunc.lower() == 'seqm':
            self.dedu_function = self.seqm
            print('self.dedu_function seqm 테스트 :', self.dedu_function)
        else:
            self.dedu_function = self.levs
            print('self.dedu_function levs 테스트 :', self.dedu_function)

    def jaro(self, cand1, cand2):
        return jellyfish.jaro_winkler(cand1, cand2 )

    def levs(self, cand1, cand2):
        return 1.-jellyfish.levenshtein_distance(cand1, cand2 ) / max(len(cand1),len(cand2))

    def seqm(self, cand1, cand2):
        return Levenshtein.ratio(cand1, cand2)



    def jacc(s1, s2):   # input: str, str
        if type(s1) and type(s2) == str:
            set1, set2 = set(s1.split()), set(s2.split())
        else:
            return print("please input the string type")

        if set1 == set2:    # 동일할 경우 1
            return 1, print("Same Sentences")

        union = set(set1).union(set(set2))  # 합집합
        # print("합집합 = ", union)
        intersection = set(set1).intersection(set(set2))    # 교집합
        # print("교집합 = ", intersection)
        jaccardScore = len(intersection)/len(union)     # 자카드 유사도
        # print("자카드 유사도 = ", jaccardScore)

        return jaccardScore

    def isReDuplicated(dataset):    # input: list(tuple)

        copySet = dataset

        for x in dataset:
            for y in dataset:
                if x == y:
                    pass
                else:
                    score = jacc(x[0], y[0])
                    if score >= 0.5:                # "때문이라는 분석이 나왔다."
                        # print("두 문장은 유사함")     # "자존심 때문이라는 분석이"
                        try:                        # 두 문단의 유사도가 0.5
                            if x[1] > y[1]:
                                copySet.remove(x)
                            else:
                                copySet.remove(y)
                        except ValueError:
                            print("이미 지워진 문장입니다.")
        return copySet



    def extract_keywords(self, text):
        try:
            if not(len(text) > 0):
                return []

            text = text.replace('\n\t',' ')
            dc = DataCore(text=text, stopword_set=self.stopword_set, windowsSize=self.windowsSize, n=self.n)
            #print('dc.sentences_str 테스트 :',dc.sentences_str)
            dc.build_single_terms_features(features=self.features)
            dc.build_mult_terms_features(features=self.features)
            resultSet = []
            todedup = sorted([cc for cc in dc.candidates.values() if cc.isValid()], key=lambda c: c.H)

            if self.dedupLim >= 1.:
                return ([ (cand.H, cand.unique_kw) for cand in todedup])[:self.top]

            for cand in todedup:
                toadd = True
                for (h, candResult) in resultSet:
                    dist = self.dedu_function(cand.unique_kw, candResult.unique_kw)
                    # print('좌 cand, 우 candResult :',cand.unique_kw," / ",candResult.unique_kw)
                    # print('dist 테스트 :',dist)
                    if dist > self.dedupLim:
                        toadd = False
                        break
                if toadd:
                    resultSet.append( (cand.H, cand) )
                if len(resultSet) == self.top:
                    break

            return isReDuplicated([(cand.kw,h) for (h,cand) in resultSet])
        except Exception as e:
            print(f"Warning! Exception: {e} generated by the following text: '{text}' ")
            return []
