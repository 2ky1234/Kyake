# -*- coding: utf-8 -*-
from pke_u_i9.yake import YAKE

def extract_keywords(text, lang="ko"):

    extractor =YAKE()

    extractor.load_document(text,language=lang)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    output=extractor.get_n_best()

    return output



