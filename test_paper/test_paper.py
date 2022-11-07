import os
import yake
from nltk.stem.porter import *
import pandas as pd


#"C:/Users/BIGTREE-HIG/Downloads/ake-datasets-master/ake-datasets-master/datasets/500N-KPCrowd/src/marujo-data/"
#"C:/Users/BIGTREE-HIG/Downloads/ake-datasets-master/ake-datasets-master/datasets/500N-KPCrowd/src/marujo-data/"

def download_file(folder_path):
    print(folder_path)
    article_file_path = []
    for test_train in ["test", "train"]:
        dir_path = folder_path + test_train + "/"  # 확장자만 바꿔주면 댐!

        for (root, directories, files) in os.walk(dir_path):
            for file in files:
                if '.txt' in file:
                    file_path = os.path.join(root, file)
                    article_file_path.append(file_path)
    print(article_file_path)


    human_file_path=[]
    for test_train in ["test", "train"]:
        dir_path = folder_path + test_train+"/" # 확장자만 바꿔주면 댐!

        for (root, directories, files) in os.walk(dir_path):
            for file in files:
                if '.key' in file:
                    file_path = os.path.join(root, file)
                    human_file_path.append(file_path)
    print(human_file_path)



    return article_file_path, human_file_path




def human_def(count,human_file_path):

    file_path = human_file_path[count]

    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()

    human_text = [line.rstrip('\n') for line in lines]


    return human_text



def article_def(count,article_file_path):

    file_path = article_file_path[count]

    with open(file_path,encoding='utf-8' ) as f:
        lines = f.readlines()

    input_text = [line.rstrip('\n') for line in lines]

    article_text=" ".join(input_text)


    return article_text



def yake_def(count,article_file_path, yake_n_gram=3):
    import yake

    text = article_def(count,article_file_path)
    language = "en"
    max_ngram_size = yake_n_gram #3
    deduplication_threshold = 0.9 #0.9
    deduplication_algo = 'seqm'
    windowSize = 1 #1
    numOfKeywords = 30 #30

    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                                features=None)
    keywords = custom_kw_extractor.extract_keywords(text)

    yake_top10=[]
    for kw in range(0,len(keywords),1):
        if kw<10:
            yake_top10.append(keywords[kw][0])

    return yake_top10


def confusion_matrix_df(output,num):

    confusion_matrix=[]

    predict = output.at[num,'yake_top10']
    answer = output.at[num,'human_text']

    stemmer = PorterStemmer()


    answer_stem = [stemmer.stem(plural) for plural in answer]
    predict_stem = [stemmer.stem(plural) for plural in predict]

    intersection = list(set(answer_stem) & set(predict_stem))

    TP = len(intersection)
    FP = len(predict_stem)-len(intersection)
    FN = len(answer_stem)-len(intersection)

    confusion_matrix.append(TP)
    confusion_matrix.append(FP)
    confusion_matrix.append(FN)

    return confusion_matrix


def score_df(output,count):

    confusion_matrix_list=[]
    for i in range(0,count,1):
        out_list=confusion_matrix_df(output,i)
        confusion_matrix_list.append(out_list)


    numerator_P=0
    denominator_P=0
    for i in range(0,count,1):
        numerator_P+=confusion_matrix_list[i][0]
        denominator_P+=(confusion_matrix_list[i][0]+confusion_matrix_list[i][1])

    micro_Precision=numerator_P/denominator_P

    numerator_R=0
    denominator_R=0
    for i in range(0,count,1):
        numerator_R+=confusion_matrix_list[i][0]
        denominator_R+=(confusion_matrix_list[i][0]+confusion_matrix_list[i][2])

    micro_Recall=numerator_R/denominator_R



    Micro_F1_Score=2*(micro_Precision*micro_Recall)/(micro_Precision+micro_Recall)

    return Micro_F1_Score



def nlp_test_step(path,yake_n_gram=3):

    folder_path=path

    article_file_path, human_file_path  = download_file(folder_path)

    article_factor=[]
    human_text_factor=[]
    yake_top10_factor=[]

    for i in range(0,len(article_file_path),1):
        article_factor.append(article_def(i,article_file_path))
        human_text_factor.append(human_def(i,human_file_path))
        yake_top10_factor.append(yake_def(i,article_file_path))

    columns={
        "article": article_factor,
        "human_text":human_text_factor,
         "yake_top10":yake_top10_factor,
    }

    output= pd.DataFrame(columns)

    return score_df(output, len(article_factor)), output

