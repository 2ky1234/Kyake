# -*- coding: utf-8 -*-
# Author: Florian Boudin and Vítor Mangaravite
# Date: 09-10-2018

"""YAKE keyphrase extraction model.
Statistical approach to keyphrase extraction described in:
* Ricardo Campos, Vítor Mangaravite, Arian Pasquali, Alípio Mário Jorge,
  Célia Nunes and Adam Jatowt.
  YAKE! Keyword extraction from single documents using multiple local features.
  *Information Sciences*, pages 257-289, 2020.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import string
from collections import defaultdict

import numpy
from nltk.metrics import edit_distance

from pke_u_i9.base import LoadFile


class YAKE(LoadFile):
    """YAKE keyphrase extraction model.
    Parameterized example::
        import pke
        from nltk.corpus import stopwords
        # 1. create a YAKE extractor.
        extractor = pke.unsupervised.YAKE()
        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)
        # 3. select {1-3}-grams not containing punctuation marks and not
        #    beginning/ending with a stopword as candidates.
        stoplist = stopwords.words('english')
        extractor.candidate_selection(n=3, stoplist=stoplist)
        # 4. weight the candidates using YAKE weighting scheme, a window (in
        #    words) for computing left/right contexts can be specified.
        window = 2
        use_stems = False # use stems instead of words for weighting
        extractor.candidate_weighting(window=window,
                                      stoplist=stoplist,
                                      use_stems=use_stems)
        # 5. get the 10-highest scored candidates as keyphrases.
        #    redundant keyphrases are removed from the output using levenshtein
        #    distance and a threshold.
        threshold = 0.8
        keyphrases = extractor.get_n_best(n=10, threshold=threshold)
    """




    def __init__(self):
        """Redefining initializer for YAKE.
        """

        super(YAKE, self).__init__()

        self.words = defaultdict(set)
        """ Container for the vocabulary. """

        self.contexts = defaultdict(lambda: ([], []))
        """ Container for word contexts. """

        self.features = defaultdict(dict)
        """ Container for word features. """

        self.surface_to_lexical = {}
        """ Mapping from surface form to lexical form. """




    def candidate_selection(self, n=3, stoplist=None, **kwargs):# <------------------------------------------------------->n-gram 길이는 여기서 수정 한다.
        """Select 1-3 grams as keyphrase candidates. Candidates beginning or
        ending with a stopword are filtered out. Words that do not contain
        at least one alpha-numeric character are not allowed.
        Args:
            n (int): the n-gram length, defaults to 3.
            stoplist (list): the stoplist for filtering candidates, defaults to
                the nltk stoplist.---------------------------------------------------------------------------------------> stoplist의 기본값은 nltk stoplist이다.
        """



        # select ngrams from 1 to 3 grams ///// -------------------------------------------------------------------------> 1. n-gram 진행하여 단어 추출

        self.ngram_selection(n=n)


        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]






        # filter candidates containing punctuation marks ///// ----------------------------------------------------------> 2. 필터링 과정

        self.candidate_filtering(stoplist=list(string.punctuation))

        for k in list(self.candidates):
            # get the candidate
            v = self.candidates[k]






        # initialize empty list if stoplist is not provided
        if stoplist is None:
            stoplist = self.stoplist




        # further filter candidates

        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # filter candidates starting/ending with a stopword or containing -------------------------------------------> 불용어로 시작/종료하거나
            # a first/last word with less than 3 characters -------------------------------------------------------------> 3글자 미만의 첫 번쩨 / 마지막 단어를 포함하는 단어 필터


            if v.surface_forms[0][0].lower() in stoplist or v.surface_forms[0][-1].lower() in stoplist \
                    or len(v.surface_forms[0][0]) < 3 \
                    or len(v.surface_forms[0][-1]) < 3:
                # v.surface_forms 맨 앞 또는 맨 뒤의 단어가 stoplist에 포함되거나 맨 앞  또는 맨 뒤의 단어가 길이가 3글자 미만이면~~~

                del self.candidates[k]
                v = self.candidates[k]


        for k in list(self.candidates):
            # get the candidate
            v = self.candidates[k]




    def _vocabulary_building(self, use_stems=False): # 단어 딕셔너리를 만듬
        """Build the vocabulary that will be used to weight candidates. Only
        words containing at least one alpha-numeric character are kept.
        Args:
            use_stems (bool): whether to use stems instead of lowercase words
                for weighting, defaults to False.
        """

        # loop through sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])
            # 첫번 째 문장에서 가장 i 번째 문장까지의 단어길이

            # loop through words in sentence
            for j, word in enumerate(sentence.words):

                # consider words containing at least one alpha-numeric character
                if self._is_alphanum(word) and \
                        not re.search('(?i)^-[lr][rcs]b-$', word):

                    # get the word or stem
                    index = word.lower()
                    if use_stems:
                        index = sentence.stems[j]

                    # add the word occurrence
                    self.words[index].add((shift + j, shift, i, word))


            # 첫번 째 문장에서 가장 i 번째 문장까지의 단어길이



    def _contexts_building(self, use_stems=False, window=2):
        """Build the contexts of the words for computing the relatedness
        feature. Words that occur within a window of n words are considered as
        context words. Only words co-occurring in a block (sequence of words
        that appear in the vocabulary) are considered.
        Args:
            use_stems (bool): whether to use stems instead of lowercase words
                for weighting, defaults to False.
            window (int): the size in words of the window used for computing
                co-occurrence counts, defaults to 2.
        """

        # loop through sentences
        for i, sentence in enumerate(self.sentences):

            # lowercase the words
            words = [w.lower() for w in sentence.words]


            # replace with stems if needed
            if use_stems:
                words = sentence.stems

            # block container
            block = []

            # loop through words in sentence
            for j, word in enumerate(words):

                # skip and flush block if word is not in vocabulary
                if word not in self.words:
                    block = []
                    continue
                # add the left context

                self.contexts[word][0].extend(
                    [w for w in block[max(0, len(block) - window):len(block)]]
                )

                # add the right context

                for w in block[max(0, len(block) - window):len(block)]:

                    self.contexts[w][1].append(word)

                # add word to the current block
                block.append(word)









    def _feature_extraction(self, stoplist=None):
        """Compute the weight of individual words using the following five
        features:
            1. CASING: gives importance to acronyms or words starting with a
               capital letter.
               CASING(w) = max(TF(U(w)), TF(A(w))) / (1 + log(TF(w)))
               with TF(U(w) being the # times the word starts with an uppercase
               letter, excepts beginning of sentences. TF(A(w)) is the # times
               the word is marked as an acronym.
            2. POSITION: gives importance to words occurring at the beginning of
               the document.
               POSITION(w) = log( log( 3 + Median(Sen(w)) ) )
               with Sen(w) contains the position of the sentences where w
               occurs.
            3. FREQUENCY: gives importance to frequent words.
               FREQUENCY(w) = TF(w) / ( MEAN_TF + STD_TF)
               with MEAN_TF and STD_TF computed on valid_tfs which are words
               that are not stopwords.
            4. RELATEDNESS: gives importance to words that do not have the
               characteristics of stopwords.
               RELATEDNESS(w) = 1 + (WR+WL)*(TF(w)/MAX_TF) + PL + PR
            5. DIFFERENT: gives importance to words that occurs in multiple
               sentences.
               DIFFERENT(w) = SF(w) / # sentences
               with SF(w) being the sentence frequency of word w.
        Args:
            stoplist (list): the stoplist for filtering candidates, defaults to
                the nltk stoplist.
        """

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = self.stoplist

        # get the Term Frequency of each word

        TF = [len(self.words[w]) for w in self.words]


        # get the Term Frequency of non-stop words
        TF_nsw = [len(self.words[w]) for w in self.words if w not in stoplist]


        # compute statistics
        mean_TF = numpy.mean(TF_nsw)
        std_TF = numpy.std(TF_nsw)
        max_TF = max(TF)

        # Loop through the words
        for word in self.words:

            # Indicating whether the word is a stopword (vitordouzi change) ---------------------------------------------> 글자 길이 제한
            self.features[word]['isstop'] = word in stoplist or len(word) < 3

            # Term Frequency
            self.features[word]['TF'] = len(self.words[word])

            # Uppercase/Acronym Term Frequencies ------------------------------------------------------------------------> 약술어, 대문자 count
            self.features[word]['TF_A'] = 0
            self.features[word]['TF_U'] = 0
            for (offset, shift, sent_id, surface_form) in self.words[word]:
                if surface_form.isupper() and len(word) > 1:
                    self.features[word]['TF_A'] += 1
                elif surface_form[0].isupper() and offset != shift:
                    self.features[word]['TF_U'] += 1




            # 1. CASING feature ///// Tcase : 약어표시와 대문자표시 단어에 가산점
            self.features[word]['CASING'] = max(self.features[word]['TF_A'],
                                                self.features[word]['TF_U'])
            self.features[word]['CASING'] /= 1.0 + math.log(
                self.features[word]['TF'])



            # 2. POSITION feature ///// Tposition : 문서 앞부분에 포함된 문장에 속한 단어일수록 가산점
            sentence_ids = list(set([t[2] for t in self.words[word]]))
            self.features[word]['POSITION'] = math.log(
                3.0 + numpy.median(sentence_ids))
            self.features[word]['POSITION'] = math.log(
                self.features[word]['POSITION'])



            # 3. FREQUENCY feature ///// TFnorm : 해당단어의 출현빈도를 정규화한값 ( 많을수록 가산점 )
            self.features[word]['FREQUENCY'] = self.features[word]['TF']
            self.features[word]['FREQUENCY'] /= (mean_TF + std_TF)



            # 4. RELATEDNESS feature ///// Trel : 주변에 등장하는 단어가 중복되지않고 다양할수록 패널티를 부여
            self.features[word]['WL'] = 0.0 # DL
            if len(self.contexts[word][0]):
                self.features[word]['WL'] = len(set(self.contexts[word][0]))
                self.features[word]['WL'] /= len(self.contexts[word][0])
            self.features[word]['PL'] = len(set(self.contexts[word][0])) / max_TF

            self.features[word]['WR'] = 0.0 # DR
            if len(self.contexts[word][1]):
                self.features[word]['WR'] = len(set(self.contexts[word][1]))
                self.features[word]['WR'] /= len(self.contexts[word][1])
            self.features[word]['PR'] = len(set(self.contexts[word][1])) / max_TF

            self.features[word]['RELATEDNESS'] = 1
            #self.features[word]['RELATEDNESS'] += self.features[word]['PL']
            #self.features[word]['RELATEDNESS'] += self.features[word]['PR']
            self.features[word]['RELATEDNESS'] += (self.features[word]['WR'] +
                                                   self.features[word]['WL']) * \
                                                  (self.features[word]['TF'] / max_TF)



            # 5. DIFFERENT feature ///// Tsentence : 해당단어가 포함된 문장이 많을수록 가산점 ( DF개념 )
            self.features[word]['DIFFERENT'] = len(set(sentence_ids))
            self.features[word]['DIFFERENT'] /= len(self.sentences)




            # assemble the features to weight words ///// formalizes this process S(t):
            A = self.features[word]['CASING']
            B = self.features[word]['POSITION']
            C = self.features[word]['FREQUENCY']
            D = self.features[word]['RELATEDNESS']
            E = self.features[word]['DIFFERENT']
            self.features[word]['weight'] = (D * B) / (A + (C / D) + (E / D))




    def candidate_weighting(self, window=2, stoplist=None, use_stems=False):  # Algorithm 5.2 Candidate Keyword socre procedure
        """Candidate weight calculation as described in the YAKE paper.
        Args:
            stoplist (list): the stoplist for filtering candidates, defaults to
                the nltk stoplist.
            use_stems (bool): whether to use stems instead of lowercase words
                for weighting, defaults to False.
            window (int): the size in words of the window used for computing
                co-occurrence counts, defaults to 2.
        """
        if not self.candidates:
            return



        # build the vocabulary
        # 단어 딕셔너리를 만든다. 글에 존재하는 해당 단어의 위치와 갯수를 알수 있다.
        self._vocabulary_building(use_stems=use_stems)

        # extract the contexts
        # 글에 존재하는 해당 단어로 부터 앞에 존재하는 단어와 뒤에 존재하는 단어들을 알 수 있다.
        self._contexts_building(use_stems=use_stems, window=window)


        # compute the word features
        # 단어의 가중치를 계산한다.
        self._feature_extraction(stoplist=stoplist)

        # compute candidate weights
        for k, v in self.candidates.items():


            # use stems
            if use_stems:

                weights = [self.features[t]['weight'] for t in v.lexical_form]
                self.weights[k] = numpy.prod(weights)
                self.weights[k] /= len(v.offsets) * (1 + sum(weights))

            # use words   ////
            else:

                lowercase_forms = [' '.join(t).lower() for t in v.surface_forms]
                for i, candidate in enumerate(lowercase_forms):
                    TF = lowercase_forms.count(candidate)


                    # computing differentiated weights for words and stopwords
                    # (vitordouzi change)
                    tokens = [t.lower() for t in v.surface_forms[i]]
                    prod_ = 1.
                    sum_ = 0.
                    for j, token in enumerate(tokens):
                        if self.features[token]['isstop']: # ------------------------------------------------------------>stop word가 포함된 경우 (O)
                            term_stop = token
                            prob_t1 = prob_t2 = 0
                            if j - 1 >= 0:
                                term_left = tokens[j-1]
                                prob_t1 = self.contexts[term_left][1].count(
                                    term_stop) / self.features[term_left]['TF']
                            if j + 1 < len(tokens):
                                term_right = tokens[j+1]
                                prob_t2 = self.contexts[term_stop][0].count(
                                    term_right) / self.features[term_right]['TF']

                            prob = prob_t1 * prob_t2
                            prod_ *= (1 + (1 - prob))
                            sum_ -= (1 - prob)
                        else: # ----------------------------------------------------------------------------------------->stop word가 포함되지 않는 경우 (X)
                            prod_ *= self.features[token]['weight']
                            sum_ += self.features[token]['weight']
                    if sum_ == -1:
                        # The candidate is a one token stopword at the start or
                        #  the end of the sentence
                        # Setting sum_ to -1+eps so 1+sum_ != 0
                        sum_ = -0.99999999999
                    self.weights[candidate] = prod_
                    self.weights[candidate] /= TF * (1 + sum_) #---------------------------------------------------------> 최종 계산
                    self.surface_to_lexical[candidate] = k

                    # weights = [self.features[t.lower()]['weight'] for t
                    #          in v.surface_forms[i]]
                    # self.weights[candidate] = numpy.prod(weights)
                    # self.weights[candidate] /= TF * (1 + sum(weights))
                    # self.surface_to_lexical[candidate] = k






    def is_redundant(self, candidate, prev, threshold=0.8): # ------------------------------------------------------------>Levenshtein 거리 (임계값을 0.8로 하였음)
        """Test if one candidate is redundant with respect to a list of already
        selected candidates. A candidate is considered redundant if its
        levenshtein distance, with another candidate that is ranked higher in
        the list, is greater than a threshold.
        Args:
            candidate (str): the lexical form of the candidate.
            prev (list): the list of already selected candidates.
            threshold (float): the threshold used when computing the
                levenshtein distance, defaults to 0.8.
        """

        # loop through the already selected candidates
        for prev_candidate in prev:
            dist = edit_distance(candidate, prev_candidate)
            dist /= max(len(candidate), len(prev_candidate))
            if (1.0 - dist) > threshold:
                return True
        return False






    def get_n_best(self,
                   n=30,
                   redundancy_removal=True,
                   stemming=False,
                   threshold=0.8): # ------------------------------------------------------------------------------------> final = S(kw) +Levenshtein 거리
                                   # 마지막에 출력할때 이용하는 함수---------------------------> n은 몇 순위까지 출력할지에 대한 갯수이다.
        """ Returns the n-best candidates given the weights.
            Args:
                n (int): the number of candidates, defaults to 10.
                redundancy_removal (bool): whether redundant keyphrases are
                    filtered out from the n-best list using levenshtein
                    distance, defaults to True.
                stemming (bool): whether to extract stems or surface forms
                    (lowercased, first occurring form of candidate), default to
                    stems.
                threshold (float): the threshold used when computing the
                    levenshtein distance, defaults to 0.8.
        """

        # sort candidates by ascending weight
        best = sorted(self.weights, key=self.weights.get, reverse=False)

        # remove redundant candidates
        if redundancy_removal:

            # initialize a new container for non redundant candidates
            non_redundant_best = []

            # loop through the best candidates
            for candidate in best:



                # test wether candidate is redundant
                if self.is_redundant(candidate,
                                     non_redundant_best,
                                     threshold=threshold):
                    continue

                # add the candidate otherwise
                non_redundant_best.append(candidate)

                # break computation if the n-best are found
                if len(non_redundant_best) >= n:
                    break

            # copy non redundant candidates in best container
            best = non_redundant_best

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        if stemming:
            for i, (candidate, weight) in enumerate(n_best):

                if candidate not in self.candidates:
                    candidate = self.surface_to_lexical[candidate]

                candidate = ' '.join(self.candidates[candidate].lexical_form)
                n_best[i] = (candidate, weight)

        # return the list of best candidates

        return n_best