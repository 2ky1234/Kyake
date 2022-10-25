#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

import os
import sys
import json
import logging
import xml.etree.ElementTree as etree
import \
    spacy  # -----------------------------------------------------------------------------------------------------------> spacy
from spacy import language
from nltk import sent_tokenize, \
    word_tokenize  # --------------------------------------------------------------------------> nltk
from spacy.util import working_dir

from pke_u_i9.data_structures import Document

if sys.platform == 'win32':
    from eunjeon import \
        Mecab  # ----------------------------------------------------------------------------------------->  eunjeon / Mecab (윈도우)
else:
    from konlpy.tag import \
        Mecab  # -----------------------------------------------------------------------------------------> konlpy.tag / Mecab (리눅스)


class Reader(object):
    def read(self, path):
        raise NotImplementedError


########################################################################################################################

class MinimalCoreNLPReader(Reader):  ###### XML Parser하는 부분
    """Minimal CoreNLP XML Parser."""

    def __init__(self):
        self.parser = etree.XMLParser()

    def read(self, path, **kwargs):
        sentences = []
        tree = etree.parse(path, self.parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in
                      sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in
                    sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            sentences.append({
                "words": [u.text for u in
                          sentence.iterfind("tokens/token/word")],
                "lemmas": [u.text for u in
                           sentence.iterfind("tokens/token/lemma")],
                "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets": [(starts[k], ends[k]) for k in
                                 range(len(starts))]
            })
            sentences[-1].update(sentence.attrib)

        doc = Document.from_sentences(sentences, input_file=path, **kwargs)

        return doc


########################################################################################################################

# FIX
def fix_spacy_for_french(nlp):  # "프랑스어"에 대한 spacy 수정
    """Fixes https://github.com/boudinfl/pke/issues/115.
    For some special tokenisation cases, spacy do not assign a `pos` field.

    해석 : 일부 특수한 토큰화의 경우 spacey는 'pos' 필드를 할당하지 않습니다.

    Taken from https://github.com/explosion/spaCy/issues/5179.
    """

    print(nlp)
    from spacy.symbols import TAG

    if nlp.lang != 'fr':
        # Only fix french model
        return nlp

    if '' not in [t.pos_ for t in nlp('est-ce')]:
        # If the bug does not happen do nothing
        return nlp

    rules = nlp.Defaults.tokenizer_exceptions

    for orth, token_dicts in rules.items():
        for token_dict in token_dicts:
            if TAG in token_dict:
                del token_dict[TAG]

    try:
        nlp.tokenizer = nlp.Defaults.create_tokenizer(nlp)  # this property assignment flushes the cache
    except Exception as e:
        # There was a problem fallback on using `pos = token.pos_ or token.tag_`
        ()

    return nlp


def list_linked_spacy_models():  # str2spacy / 보류
    """ Read SPACY/data and return a list of link_name """
    spacy_data = os.path.join(spacy.info(silent=True)['Location'], 'data')
    linked = [d for d in os.listdir(spacy_data) if os.path.islink(os.path.join(spacy_data, d))]
    # linked = [os.path.join(spacy_data, d) for d in os.listdir(spacy_data)]
    # linked = {os.readlink(d): os.path.basename(d) for d in linked if os.path.islink(d)}
    return linked


def list_downloaded_spacy_models():  # str2spacy / 보류
    """ Scan PYTHONPATH to find spacy models """
    models = []
    # For each directory in PYTHONPATH
    paths = [p for p in sys.path if os.path.isdir(p)]
    for site_package_dir in paths:
        # For each module
        modules = [os.path.join(site_package_dir, m) for m in os.listdir(site_package_dir)]
        modules = [m for m in modules if os.path.isdir(m)]
        for module_dir in modules:
            if 'meta.json' in os.listdir(module_dir):
                # Ensure the package we're in is a spacy model
                meta_path = os.path.join(module_dir, 'meta.json')
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('parent_package', '') == 'spacy':
                    models.append(module_dir)
    return models


def str2spacy(model):  # model == 영어 or 다른언어
    if int(spacy.__version__.split('.')[0]) < 3:
        downloaded_models = [os.path.basename(m) for m in list_downloaded_spacy_models()]
        links = list_linked_spacy_models()
    else:
        # As of spacy v3, links do not exist anymore and it is simpler to get a list of
        # downloaded models
        downloaded_models = list(spacy.info()['pipelines'])
        links = []
    filtered_downloaded = [m for m in downloaded_models if m[:2] == model]

    if model in downloaded_models + links:
        # Check whether `model` is the name of a model/link
        return model
    elif filtered_downloaded:
        # Check whether `model` is a lang code and corresponds to a downloaded model
        return filtered_downloaded[0]
    else:
        # Return asked model to have an informative error.
        return model


########################################################################################################################


class RawTextReader(Reader):  # raw text 리더 (언어에 따라 파서)
    """Reader for raw text."""

    def __init__(self, language=None):
        """Constructor for RawTextReader.
        Args:
            language (str): language of text to process.
        """

        self.language = language

        if language is None:  # language가 존재하지 않으면 기본적으로 영어를 적용
            self.language = 'en'

    def read(self, text, **kwargs):
        """Read the input file and use spacy to pre-process.
        Spacy model selection: By default this function will load the spacy
        model that is closest to the `language` parameter ('fr' language will
        load the spacy model linked to 'fr' or any 'fr_core_web_*' available
        model). In order to select the model that will be used please provide a
        preloaded model via the `spacy_model` parameter, or link the model you
        wish to use to the corresponding language code
        `python3 -m spacy link spacy_model lang_code`.
        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy (for spacy<3 compatibility, as of spacy v3 long texts
                should be splitted in smaller portions), default to
                1,000,000 characters (1mb).
            spacy_model (model): an already loaded spacy model.
        """


        #########################
        # 다른언어 : 영어 외의 언어 #
        ########################

        spacy_model = kwargs.get('spacy_model', None)  # 사용할 모델을 선택하기 위해 `spacy_model` 매개변수를 통해 사전 로드된 모델을 선택
        sentences = []

        if self.language != 'ko':  # ----------------------------------------------------------------------------------------------------->   English !!!!!!!!!!!!!!!!!
            if spacy_model is None:  # 사전에 로드된 spacy model과 해당 language code 가 연결이 되어 있지 않다면
                try:  # 다른언어에 대한 spacy model이 존재 한다면
                    spacy_model = spacy.load(str2spacy(self.language),
                                             disable=['ner', 'textcat', 'parser'])  # 다른언어에 대한 spacy 모델 load
                except OSError:  # 다른언어에 대한 spacy model이 존재하지 않는다면
                    logging.warning('No spacy model for \'{}\' language.'.format(self.language))
                    logging.warning('Falling back to using english model. There might '
                                    'be tokenization and postagging errors. A list of available '
                                    'spacy model is available at https://spacy.io/models.'.format(
                        self.language))
                    spacy_model = spacy.load(str2spacy('en'),  # 영어에 대한 spacy 모델 load
                                             disable=['ner', 'textcat', 'parser'])  #

                # 버전에 따른 조건문
                # spacy__version__  => "3.4.1"
                # spacy__version__.split(".") => ["3","4","1]
                # spacy__version__.split(".")[0] => "3"
                # int(spacy__version__.split(".")[0]) => 3

                if int(spacy.__version__.split('.')[0]) < 3:  # spacy버전이 3.0.0 미만
                    sentencizer = spacy_model.create_pipe(
                        'sentencizer')  # <= 'sentencizer' 파이프라인 구성 요소를 "만들어서" sentencizer변수에 입력
                else:  # 버전이 3.0.0 이상
                    sentencizer = 'sentencizer'  # <= 'sentencizer' 파이프라인 구성요소 sentencizer변수에 입력

                spacy_model.add_pipe(sentencizer)  # <= 'sentencizer' 구성요소 파이프라인에 추가

                if 'max_length' in kwargs and kwargs['max_length']:  # <= ???????
                    spacy_model.max_length = kwargs['max_length']

            spacy_model = fix_spacy_for_french(spacy_model)  # 프랑스어의 경우에 대한 spacy 수정
            spacy_doc = spacy_model(
                text)  # <------------------------------------------------------------------------------------------------------English일 경우 spacy 모델을 사용

            for sentence_id, sentence in enumerate(spacy_doc.sents):  # spacy를 이용하여 문장 분류
                # sentence_id : 문장 index
                # sentence : 문장 text

                sentences.append({
                    # ------------------------------------------------> 초기의 sentences 값은 빈값이다. / spacy를 이용하여 "단어 분류", "품사태깅"을 한다.
                    "words": [token.text for token in sentence],  # 입력된 단어 그대로 가져온다.
                    "lemmas": [token.lemma_ for token in sentence],  # 매칭된 단어의 원형을 가져온다.
                    # FIX : This is a fallback if `fix_spacy_for_french` does not work
                    "POS": [token.pos_ or token.tag_ for token in sentence],  # 매칭된 단어의 품사를 가져온다.
                    "char_offsets": [(token.idx, token.idx + len(token.text)) for token in sentence]
                    # 문장에서의 단어 길이의 범위 / 단어 : (시작, 끝)
                    # ex) 'PRAGUE—European' => "PRAGUE" : (0,6)
                    # "—" : (6,7)
                    # "European" : (7, 15)

                })
            for i in range(0, len(sentences), 1):
                sentences_2 = sentences[i]


        else:  # ----------------------------------------------------------------------------------------------------------------------------->   korean !!!!!!!!!!!!!!!!!
            tag_set = {'NNP': 'NOUN', 'NNG': 'NOUN', 'NNB': 'AUX', 'NNBC': 'AUX', 'NR': 'NUM', 'NP': 'PROPN',
                       'VV': 'VERB', 'VA': 'ADJ', 'VX': 'PART', 'VCP': 'PART', 'VCN': 'PART',
                       'MM': 'DET', 'MAG': 'ADV', 'MAJ': 'CONJ', 'IC': 'X',
                       'JKS': 'PART', 'JKC': 'PART', 'JKG': 'PART', 'JKO': 'PART', 'JKB': 'PART', 'JKV': 'PART',
                       'JKQ': 'PART', 'JC': 'PART', 'JX': 'PART',
                       'EP': 'PART', 'EF': 'PART', 'EC': 'PART', 'ETN': 'PART', 'ETM': 'PART',
                       'XPN': 'PART', 'XSN': 'PART', 'XSV': 'PART', 'XSA': 'PART', 'XR': 'PART',
                       'SF': 'PUNCT', 'SE': 'PUNCT', 'SSO': 'PUNCT', 'SSC': 'PUNCT', 'SC': 'PUNCT', 'SY': 'PUNCT',
                       'SH': 'NOUN', 'SL': 'NOUN', 'SN': 'NUM'}

            spacy_doc = Mecab()  # <------------------------------------------------------------------------------------------------------ korean일 경우 Mecab 모델을 사용

            text = text.replace('·', ',').replace('+', ',')  # 문장에 존재하는 NLT를 ','로 바꿈

            text

            for sentence_id, sentence in enumerate(sent_tokenize(text)):  # nltk를 이용하여 문장 분류

                # sentence_id : 문장 index
                # sentence : 문장

                words_list, lemmas_list, pos_list, char_offsets_list = [], [], [], []
                offset = 0
                """"
                word_tokenize_add_list = []
                for i in range(0, len(word_tokenize(sentence)), 1):
                    word_tokenize_add_list.append = word_tokenize(sentence)[i] + "는"
                """

                for word_idx, words in enumerate(word_tokenize(sentence)):  # nltk를 이용하여 단어 분류

                    # word_idx : 문장에서의 단어 index
                    # words : 단어 text

                    pos_by_word = spacy_doc.pos(words + ' .')[:-1]  # Macab을 이용한 단어 품사 태킹
                    # spacy_doc.pos(words+' .') => [('프라하', 'NNP'), ('.', 'SF')]
                    # spacy_doc.pos(words + ' .')[:-1]=>  [('프라하', 'NNP')]

                    # Forward Inflect suspected
                    # "고유 명사", "일반 명사"로 태깅된 것을 조금 더 세밀하게 확인하기 위해 다시한번 "해당 단어"와 "주변 단어"들을 이용하여 문맥 파악을 통해 다시 한번 태깅한다..

                    # ========================================================= 예시 =======================================================================

                    # 1.
                    # pos_by_word =>  [('지고', 'NNG')]
                    # pos_by_context => [('지', 'VV'), ('고', 'EC')]

                    # 2.
                    # pos_by_word =>  [('유럽연합', 'NNP')]
                    # pos_by_context => [('유럽', 'NNP'), ('연합', 'NNG')]

                    # 3.
                    # pos_by_word =>  [('접어', 'NNG'), ('들', 'XSN')]
                    # pos_by_context => [('접어들', 'VV+ETM'), ('것', 'NNB')]

                    # ============================================== 예시 =================================================================================


                    if pos_by_word[0][1] in ('NNP', 'NNG'):  # 단어의 품사가 (NNP : 고유 명사 / NNG :일반 명사) 라면~

                        if word_idx < len(word_tokenize(sentence)):  # Inflected into multiple PoS  / 단어가 해당 문장의 범위에 포함된다면~~

                            # 원본 pos_by_context = spacy_doc.pos(' '.join(word_tokenize(sentence)[word_idx:word_idx+2]))[:2]  # PoS using following the word
                            pos_by_context = spacy_doc.pos(' '.join(word_tokenize(sentence)[word_idx:word_idx + 2]))[:2]

                            #============================ 원본에서 일부 수정=======================================================
                            if len(pos_by_word)>1 and len(pos_by_context)>1: # <== spacy_pos에 [('키', 'NNG'), ('이', 'VCP'), ('우', 'EF')]이 적용되는 것을 막고 [('키', 'NNG'), ('이우', 'NNG')]을 적용하기 위해

                                if pos_by_context[0][0] != pos_by_word[0][0] or pos_by_context[1][0] != pos_by_word[1][0]:  # pos_by_context의 태깅된 단어와 pos_by_word에서의 태깅된 단어와 다르면~~~

                                    spacy_pos = pos_by_context  # "수정"된 spacy_pos 태깅된 단어와 품사 입력
                                else:  # pos_by_context의 태깅된 단어와 pos_by_word에서의 태깅된 단어와 같으면~~~

                                    # ============================ 원본에서 일부 수정=======================================================
                                    # spacy_pos에 [('우크라', 'NNP'), ('이나', 'JC'), ('군', 'NNG'), ('이', 'JKS')]가 적용되는 것을 막고
                                    # spacy_pos에 [('우크라이나', 'NNP'), ('군', 'NNG'), ('이', 'JKS')]가 적용시키기 위해
                                    word_JC=[]
                                    for i in range(0, len(pos_by_word), 1):
                                        if 'JC' in pos_by_word[i]:
                                            JC_indx = i

                                            pos_by_word.insert(JC_indx + 1, ('는', 'JX'))

                                            word_JC=pos_by_word


                                    if word_JC!=[]:

                                        word_JC_list=[]
                                        for a in range(0, len(word_JC), 1):
                                                word_JC_list.append(word_JC[a][0])

                                        word_JC_str=''.join(word_JC_list)

                                        pos_by_word_JC = spacy_doc.pos(word_JC_str)

                                        count = len(pos_by_word_JC)
                                        for i in range(0, count, 1):
                                            try:
                                                pos_by_word_JC.remove(('는', 'JX'))
                                            except ValueError:
                                                continue
                                        spacy_pos = pos_by_word_JC

                                    else:
                                        spacy_pos = pos_by_word  # "수정"된 spacy_pos 태깅된 단어와 품사 입력
                                    # =============================================================================================

                            else:  # pos_by_context의 태깅된 단어와 pos_by_word에서의 태깅된 단어와 같으면~~~

                                spacy_pos = pos_by_word  # "기존"의 spacy_pos 태깅된 단어와 품사 입력

                            # =========================================================================================

                        else:  # 단어가 해당 문장의 범위에 포함되지 않는다면 (이런 경우가 있을 수가 있나....?)

                            spacy_pos = pos_by_word


                    else:  # 품사가 (NNP : 고유 명사 / NNG :일반 명사) 아니라 라면~

                        spacy_pos = pos_by_word



                    # ============================ 원본에서 추가한 코드=======================================================
                    for i in spacy_pos:
                        # [['정치', 'NNG'], ['자금', 'NNG'], ['법', 'NNG']] 경우 두번 해줘야지 된다 -> [('정치자금', 'NNG'), ('법', 'NNG')] -> [('정치자금법', 'NNG')]
                        if len(spacy_pos)>1:

                            # ex) 바로 위의 절차를 진행한 후 spacy_pos리스트로 [('우크라이나', 'NNP'), ('군', 'NNG'), ('이', 'JKS')]가 나오는데 나중에 이를 활용하여
                            #     => TF : [['우크라이나'], ['우크라이나'], ['우크라이나'], ['우크라이나'], ['우크라이나'], ['우크라이나'], ['우크라이나'], ['우크라이나군']] => TF를 할때 "우크라이나"에 해당하는 것만 count해야하지만 "우크라이나군"까지 count한다.
                            #     이러한 상황을 막기위해 수정해 준다. -> spacy_pos[('우크라이나군', 'NNP'), ('이', 'JKS')]
                            if spacy_pos[0][1]=="NNP" and spacy_pos[1][1]=="NNG" and len(spacy_pos[1][0])>=1: # == if spacy_pos[0][1]=="NNP" and spacy_pos[1][1]=="NNG"

                                spacy_pos_list = []
                                for i in range(0, len(spacy_pos), 1):
                                    spacy_pos_list.append(list(spacy_pos[i]))

                                # NNP <= NNP+NNG and del NNG
                                spacy_pos_list[0][0] = spacy_pos_list[0][0] + spacy_pos_list[1][0]
                                spacy_pos_list.remove(spacy_pos_list[1])

                                spacy_pos_tuple = []
                                for i in range(0, len(spacy_pos_list), 1):
                                    spacy_pos_tuple.append(tuple(spacy_pos_list[i]))


                                spacy_pos=spacy_pos_tuple


                            # ex) 바로 위의 절차를 진행한 후 spacy_pos로 [('키', 'NNG'), ('이우', 'NNG')]처럼 NNG가 연속해서 나오는데 나중에 이를 활용하여
                            #     NNG+NNG 하나로 묶어서 [('키이우', 'NNG')]를 만든다.
                            elif spacy_pos[0][1] == "NNG" and spacy_pos[1][1] == "NNG" :

                                spacy_pos_list = []
                                for i in range(0, len(spacy_pos), 1):
                                    spacy_pos_list.append(list(spacy_pos[i]))

                                spacy_pos_list[0][0] = spacy_pos_list[0][0] + spacy_pos_list[1][0]
                                spacy_pos_list.remove(spacy_pos_list[1])

                                spacy_pos_tuple = []
                                for i in range(0, len(spacy_pos_list), 1):
                                    spacy_pos_tuple.append(tuple(spacy_pos_list[i]))

                                spacy_pos = spacy_pos_tuple


                            # ex) 바로 위의 절차를 진행한 후 spacy_pos리스트로 [('LG', 'SL')]가 나오는데 나중에 이를 활용하여 (6.txt이용)
                            #     => TF : [['LG전자'], ['LG전자'], ['LG전자'], ['LG']] => TF를 할때 'LG전자'에 해당하는 것만 count해야하지만 "LG"까지 count한다.
                            #     이러한 상황을 막기위해 수정해 준다. -> [('LG', 'SL'), ('전자', 'NNG'), ('가', 'JKS')] -> [('LG전자', 'NNP'), ('가', 'JKS')]
                            elif (spacy_pos[0][1] == "SL" and spacy_pos[1][1] == "NNG") or (
                                    spacy_pos[0][1] == "NNG" and spacy_pos[1][1] == "SL"):

                                spacy_pos_list = []
                                for i in range(0, len(spacy_pos), 1):
                                    spacy_pos_list.append(list(spacy_pos[i]))

                                spacy_pos_list[0][0] = spacy_pos_list[0][0] + spacy_pos_list[1][0]
                                spacy_pos_list[0][1] = "NNP"
                                spacy_pos_list.remove(spacy_pos_list[1])

                                spacy_pos_tuple = []
                                for i in range(0, len(spacy_pos_list), 1):
                                    spacy_pos_tuple.append(tuple(spacy_pos_list[i]))

                                spacy_pos = spacy_pos_tuple



                    # =====================================================================================================





                    if len(spacy_pos) > 1 and spacy_pos[0][1] in ('NNP', 'NNG', 'SL'):

                        # spacy_pos의 크기가 1 초과이고, spacy_pos에서 가장 앞에 있는 단어의 품사가
                        # (NNP : 고유 명사 / NNG:일반 명사 / SL : 한국어가 아닌 단어) 라면 ~~~~
                        try:

                            pos_offset = [pos[1] not in ('NNP', 'NNG', 'SL') for pos in spacy_pos].index(True)
                            #             pos[1] not in ('NNP','NNG','SL') for pos in spacy_pos
                            # => spacy_pos 범위의 포함된 각각 단어의 품사가 'NNP','NNG','SL'라면 False, 아니면 True

                            # pos_offset : spacy_pos리스트 범위에서 처음부터 어디까지, 단어의 품사가 ('NNP', 'NNG', 'SL')로 구성되어 있는지 위치 확인


                            # aux_offset : spacy_pos 에서 가장 마지막에 위치하는 ('NNP','NNG','SL')품사에 해당하는 단어의 길이

                            aux_offset = 0
                            aux_offset = sum([len(word[0]) for word in spacy_pos[:pos_offset]])
                            # aux_offset+=len(word[0])

                            # aux_offset = sum([len(word[0]) for word in spacy_pos[:pos_offset]])

                            # aux_offset = words.find(spacy_pos[pos_offset-1][0][-1])+1
                            # print(words,pos_offset,aux_offset)
                        except ValueError as e:  # 입력되는 타입 오류 (X), 값 오류(O)
                            # pos_offset = [pos[1] not in ('NNP', 'NNG', 'SL') for pos in spacy_pos].index(True)를 실행 시킬때
                            # [('기쿠요', 'NNP'), ('마치', 'NNG')] 처럼 spacy_pos 안에 'NNP', 'NNG', 'SL'외에 단어가 없어 True가 존재하지 않아 오류가 뜬다.
                            # 오류가 발생하므로 except에 있는 코드가 돌아가는데
                            # 원본 이었던 aux_offset = len(words)를 사용해 버리면 "키쿠요마치"만 word_list에 들어가야하지만 (words : 기쿠요마치에)가 들어가 버리기 때문에 나중에 TF를 잘못하거나, 잘못된 출력을 할 수 있어 위험하다.
                            # 그래서 [('기쿠요', 'NNP'), ('마치', 'NNG')] 리스트 안에 있는 단어들만 뽑아 오기 위해
                            # aux_offset = sum([len(word[0]) for word in spacy_pos])로 코드를 변경하였다.
                            pos_offset = 1

                            # 원본 aux_offset = len(words)
                            aux_offset = sum([len(word[0]) for word in spacy_pos])


                    else:

                        pos_offset = 1
                        aux_offset = len(words)

                    # Backward Inflect suspected
                    # if len(spacy_pos[0][0])==1 and spacy_pos[0][1]=='NNG' and pos_list:
                    # pos_list는 초기에는 빈값이다.

                    if spacy_pos[0][1] == 'NNG' and pos_list:  # spacy_pos에서 가장 앞에 있는 단어가 NNG(일반명사)이고
                        # pos_list가 빈 값이 아니라면~~~


                        if pos_list[-1] == 'NNG':  # pos_list의 가장 마지막 위치에 NNG(일반명사)가 들어온다면~~~

                            pos_with_last = spacy_doc.pos(words_list[-1] + ' ' + words)
                            # (words_list의 가장 마지막 단어 + " " + nltk로 분류한 단어)를 다시 품사 태깅하여 pos_with_last에 넣는다.

                            try:

                                last_offset = [(pos[1] == 'NNB' and len(pos[0]) == 1) for pos in pos_with_last].index(
                                    True)

                                spacy_pos = pos_with_last[last_offset:]

                            except ValueError:
                                pass
                            # last_offset = [pos[0].find(spacy_pos[0][0]) for pos in pos_with_last].index(0)
                            # spacy_pos = pos_with_last[last_offset:]



                    # ==============================================================================================================================


                    words_list.append(words[:aux_offset])

                    # lemmas_list.append(spacy_pos[0][0])

                    # Number
                    # 숫자 처리를 위해
                    if len(spacy_pos) > 1 and spacy_pos[0][1] == 'SN':  # and spacy_pos[-1][1]=='SN':


                        # words_list.append(words)
                        # nltk로 분류된 단어가 탱깅을 통해 다시 분류 되었을때 'SN'(숫자)품사가 포함되면 nltk로 분류한 단어를 그대로 입력 한다.
                        lemmas_list.append(words)




                    else:

                        # words_list.append(words[:aux_offset])
                        # spacy_pos에서 가장 앞에 있는 단어들만 입력
                        lemmas_list.append(spacy_pos[0][0])
                        # 오류! : lemmas_list.append(words[:aux_offset])

                    # if len(words)==1 and spacy_pos[0][1].startswith('N'):   # Dependent Nouns

                    # spacy_pos의 첫번째 단어의 길이가 "1"이고 품사가 "N"으로 시작한다면 ~~~

                    if len(spacy_pos[0][0]) == 1 and spacy_pos[0][1].startswith('N'):  # Dependent Nouns

                        pos_list.append('NNB')  # pos_list에 'NNB'를 넣는다.


                    # spacy_pos의 크기가 "1"초과이고 첫번째 단어의 품사가 "SN"이면
                    elif len(spacy_pos) > 1 and spacy_pos[0][1] == 'SN':  # and spacy_pos[-1][1]=='SN': # Number

                        pos_list.append('SN')


                    else:

                        replace_word = spacy_pos[0][1].replace('SL', 'NNP')

                        pos_list.append(spacy_pos[0][1].replace('SL', 'NNP'))  # 첫번째 단어의 품사를 "SL"를 "NNP"로 대처한다.


                    if aux_offset < len(words):

                        words_list.append(words[aux_offset:])

                        lemmas_list.append(spacy_pos[pos_offset][0])


                        replace_word_2 = spacy_pos[pos_offset][1].replace('SL', 'NNP')
                        pos_list.append(spacy_pos[pos_offset][1].replace('SL', 'NNP'))


                # char_offsets_list을 만드려고 하는 구조
                for word in words_list:
                    if char_offsets_list:
                        offset = char_offsets_list[-1][1] + 1
                    char_offsets_list.append((offset, offset + len(word)))


                # pos_list의 품사들이 tag_set에 있는 품사들과 매칭을 시켜 매칭이 되는 것이 있으면 매칭이 되는 것으로 변경하고
                # 매칭이 되어 있지 않는 것은 'X'를 넣어서 표현한다.

                pos_list = [tag_set[pos] if tag_set.get(pos) else 'X' for pos in pos_list]


                sentences.append({
                    "words": [word for word in words_list],
                    "lemmas": [lemma for lemma in lemmas_list],
                    "POS": [pos for pos in pos_list],
                    "char_offsets": [offset for offset in char_offsets_list]
                })

            sent_token = []
            for sentence_id, sentence in enumerate(sent_tokenize(text)):
                sent_token.append(sentence)

            for i in range(0, len(sentences), 1):
                sentences_1 = sentences[i]


        doc = Document.from_sentences(
            sentences, input_file=kwargs.get('input_file', None), **kwargs)


        return doc