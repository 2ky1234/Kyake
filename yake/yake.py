# -*- coding: utf-8 -*-

"""Main module."""

import string
import os
import jellyfish
from .Levenshtein import Levenshtein

#from .datarepresentation import DataCore
from .datarepresentation_korea import DataCore

def jacc(s1, s2):   # input: str, str
    """
    Jaccard Similarity Function.
    In this source, use 2 parameter via paragraph.
    Return value is score of Jaccard Similarity.
    """
    if type(s1) and type(s2) == str:
        set1, set2 = set(s1.split()), set(s2.split())
    else:
        return print("please input the string type")

    if set1 == set2:
        return 1    # 동일할 경우 1

    union = set(set1).union(set(set2))                  # 합집합
    intersection = set(set1).intersection(set(set2))    # 교집합
    jaccardScore = len(intersection)/len(union)         # 자카드 유사도

    return jaccardScore

def isUniqueText(dataset, WG):    # dataset: [(float, <class>), (float, <class>), ..., (float, <class>)]
    """
    Extense from Jaccard Similarity Function.
    In this function, use 2 parameter, dataset and WG(None or True or False), dataset is composed of tuple.
    Return value is set of dataset composed of tuple, except duplicated things.

    WG : None
    Function don't copy dataset, and erase words based on Jacard similarity if the target is in low rank.

    WG : False
    Function to erase words based on Jacard similarity if the target is in low rank.

    WG : True
    Function that is not cleared if it is a subdependent.
    """

    if WG is None:
        copyset = dataset
    else:
        copyset = dataset.copy()

    for x in dataset:
        if x not in copyset:
            continue
        for y in dataset:
            if x==y:
                continue
            elif y not in copyset:
                continue
            elif (' ' not in x[1].kw) or (' ' not in y[1].kw) or (len(x[1].kw.split()) < len(y[1].kw.split())):
                continue
            elif WG is False or None:                               # WG(word-gram)가 False면 (디폴트값)
                score = jacc(x[1].kw, y[1].kw)        # 자카드 유사도 실행
                if score >= 0.5:                                    # 기준 문장보다 하위 티어의 경우에만 자카드 검증 실행됨
                    try:
                        if x[1].H > y[1].H:
                            copyset.remove(x)
                        else:
                            copyset.remove(y)
                    except ValueError:
                        continue
            elif WG is True:                                        # WG(word-gram)가 True면
                if (len(x[1].kw.split()) == len(y[1].kw.split())):  # 하위 그램은 보존함
                    score = jacc(x[1].kw, y[1].kw)
                    if score >= 0.5:
                        try:
                            if x[1].H > y[1].H:
                                copyset.remove(x)
                            else:
                                copyset.remove(y)
                        except ValueError:
                            continue
    return copyset

def isReDuplicated(dataset, WG, ReDup):     # dataset: [(float, <class>), (float, <class>), ..., (float, <class>)]
    """
    ReDup : None
    Function do nothing.

    ReDup : False
    Function that erases words that belong to a duplicate when the target is in low rank.

    ReDup : True
    Function that removes a word from a data set if it contains a phrase.
    """

    if ReDup is None:
        return dataset

    if WG is None:
        copyset = dataset
    else:
        copyset = dataset.copy()

    for x in dataset:
        if x not in copyset:        # x: (float, <class>)
            continue
        for y in dataset:
            if x==y:
                continue
            elif y not in copyset:
                continue
            else:
                try:
                    amount = 0
                    for z in y[1].kw.split():
                        if (' ' not in x[1].kw) or (len(x[1].kw.split()) < len(y[1].kw.split())):
                            continue

                        if ReDup==False:
                            if x[0] > y[0]:
                                continue
                            if z in x[1].kw.split():
                                amount = amount + 1
                            if amount == len(y[1].kw.split()):
                                copyset.remove(y)

                        if ReDup==True:
                            if z in x[1].kw.split():
                                amount = amount + 1
                            if amount == len(y[1].kw.split()):
                                copyset.remove(y)
                except ValueError:
                    continue
    return copyset
        
class KeywordExtractor(object):

    def __init__(self, lan="ko", n=[1,2,3], dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, stoplen = 2, features=None, stopwords=None, WG=False, ReDup=False):
        self.lan = lan
        self.stoplen = stoplen
        dir_path = os.path.dirname(os.path.realpath(__file__))

        local_path = os.path.join("StopwordsList", "stopwords_%s.txt" % lan[:2].lower())

        if os.path.exists(os.path.join(dir_path,local_path)) == False:
            local_path = os.path.join("StopwordsList", "stopwords_noLang.txt")
        
        resource_path = os.path.join(dir_path,local_path)
        self.resource_path = resource_path  # stopword 관리를 위한 변수 추가

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
        self.WG = WG
        self.ReDup = ReDup
        self.windowsSize = windowsSize
        if dedupFunc == 'jaro_winkler' or dedupFunc == 'jaro':
            self.dedu_function = self.jaro
            # print('self.dedu_function jaro 테스트 :', self.dedu_function)
        elif dedupFunc.lower() == 'sequencematcher' or dedupFunc.lower() == 'seqm':
            self.dedu_function = self.seqm
            # print('self.dedu_function seqm 테스트 :', self.dedu_function)
        else:
            self.dedu_function = self.levs
            # print('self.dedu_function levs 테스트 :', self.dedu_function)

    def jaro(self, cand1, cand2):
        """
        using jaro_winkler distance algorithm.
        completed by original yake!
        """
        return jellyfish.jaro_winkler(cand1, cand2 )

    def levs(self, cand1, cand2):
        """
        using levenshtein_distance algorithm.
        completed by original yake!
        """
        return 1.-jellyfish.levenshtein_distance(cand1, cand2 ) / max(len(cand1),len(cand2))

    def seqm(self, cand1, cand2):
        """
        using levenshtein_distance ratio algorithm.
        completed by original yake!
        """
        return Levenshtein.ratio(cand1, cand2)

    def extract_keywords(self, text):
        """
        Function for extract keywords.
        Return value is set of list composed of tuple, except duplicated things.
        """
        try:
            if not(len(text) > 0):
                return []

            text = text.replace('\n\t',' ')
            # print('self.stopword_set 출력 :',self.stopword_set)
            dc = DataCore(text=text, stopword_set=self.stopword_set, windowsSize=self.windowsSize, n=self.n, stoplen = self.stoplen)
            # 형태소 원형 변경본 -> 원본으로 변경하는 작업 
            dc.build_single_terms_features(features=self.features)
            # 원본은 dc.initial_sentences_str에 담겨있음 
            # 형태소 원형 변경본의 경우 dc.sentences_obj의 각칸마다 들어있음  
            
            #print('dc initial 테스트 :', dc.initial_sentences_str)
            #print('candidates 테스트 :',dc.sentences_obj)#[0][0][0][2].H)
            #print('이곳을 추출')
            dc.build_mult_terms_features(features=self.features)
            resultSet = []
            todedup = sorted([cc for cc in dc.candidates.values() if cc.isValid()], key=lambda c: c.H)
            # print('todedup 테스트 :', len(todedup)) # n-gram의 composed_word 들로 들어감
            
            # 이곳에 펼친 값을 붙이자
            
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

            # stopword 자동 수정 로직
            try:
                with open(self.resource_path, 'r+', encoding='utf-8') as f:
                    stopwordset = list(set( f.read().split("\n") ))
                    stopwordset.sort()
                    f.truncate(0)
                    f.close()

                with open(self.resource_path, 'a+', encoding='utf-8') as ft:
                    for i in stopwordset:
                        ft.write(i)
                        ft.write('\n')
                    ft.close()

            except:
                print('Warning, read stopword list as ISO-8859-1')
                with open(self.resource_path, 'r+', encoding='ISO-8859-1') as f:
                    stopwordset = list(set( f.read().lower().split("\n") ))
                    stopwordset.sort()
                    f.truncate(0)
                    f.close()

                with open(self.resource_path, 'a+', encoding='utf-8') as ft:
                    for i in stopwordset:
                        ft.write(i)
                        ft.write('\n')
                    ft.close()

            FirstReturnSet = isReDuplicated(resultSet, self.WG, self.ReDup)    # [(cand.kw,h) for (h,cand) in resultSet]
            SecondReturnSet = isUniqueText(FirstReturnSet, self.WG)

            return [(cand.origin_terms, h) for (h,cand) in SecondReturnSet]

        except Exception as e:
            print(f"Warning! Exception: {e} generated by the following text: '{text}' ")
            return []

