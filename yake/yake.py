# -*- coding: utf-8 -*-

"""Main module."""

import string
import os
import jellyfish
from .Levenshtein import Levenshtein

#from .datarepresentation import DataCore
from .datarepresentation_korea import DataCore

def jacc_for_word(s1, s2):
    """
    Jaccard Similarity Function.
    In this source, use 2 parameter via word.
    Return value is score of Jaccard Similarity.
    """
    if type(s1) and type(s2) == str:
        set1, set2 = set(list(''.join(s1.split()))), set(list(''.join(s2.split())))
    else:
        return print("please input the string type")

    if set1 == set2:
        return 1    # 동일할 경우 1

    union = set(set1).union(set(set2))                  # 합집합
    intersection = set(set1).intersection(set(set2))    # 교집합
    jaccardScore = len(intersection)/len(union)         # 자카드 유사도

    return jaccardScore

def jacc_for_paragraph(s1, s2):   # input: str, str
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

def isParagraphReDuplicated(dataset, COpy):    # input: list(tuple)
    """
    Extense from Jaccard Similarity Function.
    In this function, use 2 parameter, list and (T or F). list is composed of tuple.
    Return value is set of list composed of tuple, except duplicated things.

    When COPY parameter set True, return value will minimized.
    """
    
    # dataset: [(float, <class>), (float, <class>), ..., (float, <class>)]

    if COpy is True:
        copySet = dataset.copy()
    elif COpy is False:
        copySet = dataset

    for x in dataset:                   # x: (float, <class>)
        if x not in copySet:
            continue

        for y in dataset:
            if x==y:
                continue
            elif y not in copySet:
                continue
            else:
                score = jacc_for_paragraph(x[1].kw, y[1].kw)
                if score >= 0.5:
                    try:
                        if x[1].H > y[1].H:
                            copySet.remove(x)
                        else:
                            copySet.remove(y)
                    except ValueError:
                        continue
    return copySet

def isWordReDuplicated(dataset, COpy):    # input: list(tuple)
    """
    Extense from Jaccard Similarity Function.
    In this function, use 2 parameter, list and (T or F). list is composed of tuple.
    Return value is set of list composed of tuple, except duplicated things.

    When COPY parameter set True, return value will minimized.
    """
    # dataset: [(float, <class>), (float, <class>), ..., (float, <class>)]

    if COpy is True:
        copySet = dataset.copy()
    elif COpy is False:
        copySet = dataset

    for x in dataset:                   # x: (float, <class>)
        if x not in copySet:
            continue

        for y in dataset:
            if x==y:
                continue
            elif y not in copySet:
                continue
            else:
                score = jacc_for_paragraph(x[1].kw, y[1].kw)
                if score >= 0.5:
                    try:
                        if x[1].H > y[1].H:
                            copySet.remove(x)
                        else:
                            copySet.remove(y)
                    except ValueError:
                        continue
    return copySet


def isUniqueText(dataset, COpy, Usage):
    """
    Function that removes a single word from the dataset if a phrase containing the word exists.
    Set Usage parameter True, when want to use this function.
    
    examples) '한' '한 나라' '나라의' '한 나라의 대통령' -> '한 나라' '한 나라의 대통령'
    """

    if Usage is True:
        if COpy is True:
            copySet = dataset.copy()
        elif COpy is False:
            copySet = dataset

        for x in range(len(dataset)):
            if ' ' in dataset[x][1].kw:
                continue
            for y in range(len(dataset)):
                if x == y:
                    continue
                elif ' ' not in dataset[y][1].kw:
                    continue
                else:
                    try:
                        if dataset[x][1].kw in dataset[y][1].kw.split():
                            copySet.remove(dataset[x])
                    except ValueError:
                        continue
        return copySet
    else:
        return dataset

def isUniqueSentence(dataset, COpy, Usage):
    """
    Function that deletes a phrase from the dataset if a phrase containing the phrase exists.
    Set Usage parameter True, when want to use this function.

    examples) '한 나라' '나라의' '한 나라 국민' -> '나라의' '한 나라 국민'
    """

    if Usage is True:
        if COpy is True:
            copySet = dataset.copy()
        elif COpy is False:
            copySet = dataset

        for x in range(len(dataset)):
            if ' ' not in dataset[x][1].kw:
                continue
            for y in range(len(dataset)):
                if x == y:
                    continue
                elif ' ' not in dataset[y][1].kw:
                    continue
                else:
                    try:
                        if dataset[x][1].kw in dataset[y][1].kw.split():
                            copySet.remove(dataset[x])
                    except ValueError:
                        continue
        return copySet
    else:
        return dataset
class KeywordExtractor(object):

    def __init__(self, lan="ko", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, features=None, stopwords=None, COpy=True, Usage=False):
        self.lan = lan

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
        self.COpy = COpy
        self.Usage = Usage
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
            dc = DataCore(text=text, stopword_set=self.stopword_set, windowsSize=self.windowsSize, n=self.n)
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

            # print("stopword 저장 위치: ", self.resource_path)
            # stopword 자동 수정 로직
            try:
                with open(self.resource_path, 'r+', encoding='utf-8') as f:
                    stopwordset = list(set( f.read().split("\n") ))
                    stopwordset.sort()
                    # print("stopword 리스트 test1: ", stopwordset)
                    f.truncate(0)
                    f.close()

                # print("stopword 리스트: ", stopwordset)
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
                    # print("stopword 리스트 test1: ", stopwordset)
                    f.truncate(0)
                    f.close()

                # print("stopword 리스트 test2: ", stopwordset)
                with open(self.resource_path, 'a+', encoding='utf-8') as ft:
                    for i in stopwordset:
                        ft.write(i)
                        ft.write('\n')
                    ft.close()

            # beforeReturnSet = isUniqueText(resultSet, self.COpy, self.Usage)    # [(cand.kw,h) for (h,cand) in resultSet]
            # middleReturnSet = isUniqueSentence(beforeReturnSet, self.COpy, self.Usage)
            afterReturnSet = isParagraphReDuplicated(resultSet, self.COpy)
            finalReturnSet = isWordReDuplicated(afterReturnSet, self.COpy)
            # print(beforeReturnSet)
            # print(resultSet[0][1]) # .origin_terms)

            # testset = isParagraphReDuplicated(resultSet, self.COpy)

            return [(cand.origin_terms,h) for (h,cand) in resultSet]  # [(cand.kw, h) for (h,cand) in resultSet]

        except Exception as e:
            print(f"Warning! Exception: {e} generated by the following text: '{text}' ")
            return []
        
        