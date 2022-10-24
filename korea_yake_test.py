# -*- coding: utf-8 -*-
from yake import yake
import pke_u_i9.pke_u_i9 as pke_u_i9
from yake.datarepresentation_korea import DataCore
import pandas as pd

#text = "Don't be fooled by the dark fool sounding name. Mr. Jone's Orphange is as cheery goes for a pastry shop. My dogs sounded is cute and My dog sound is not cute" 
#text = "hello world i am a boy. I want to go ride a car. dog is crazy and crazy dog barking"
#text =  "Google is acquiring data science community Kaggle. Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague , but given that Google is hosting its Cloud Next conference in San Francisco this week, the official announcement could come as early as tomorrow. Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom and Ben Hamner in 2010. The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science and machine learning competitions. With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) since its launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, Google chief economist Hal Varian, Khosla Ventures and Yuri Milner"
#text = "러시아가 우크라이나 키이우 등을 공습한 것은 군 내부 비판과 블라디미르 푸틴 러시아 대통령의 자존심 때문이라는 분석이 나왔다. 영국 가디언지는 10일(현지시간) 푸틴 대통령의 대규모 공습은 국내 군 비판세력, 러시아가 침공에서 실패하고 있다는 사실, 크림대교 폭발 후 상처받은 자존심에 대한 절박한 답변이라고 풀이했다. 가디언에 따르면 미국 싱크탱크 카네기국제평화재단의 선임 연구원 안드레이 콜레스니코프는 '지금 푸틴이 하는 것은 사소한 복수'라며 '개인적 복수도 있다'고 말했다. 러시아 전쟁 전문가와 군사 블로거 등은 수개월간 우크라이나를 상대로 전면전을 벌이라고 촉구해왔고 키이우 등의 거리에 시신이 있는 끔찍한 사진이 나오자 지금은 만족해한다. 최근 러시아군 수뇌부를 비판했던 람잔 카디로프 체첸 공화국 수장은 '(볼로디미르) 젤렌스키(우크라이나 대통령), 우리는 러시아가 아직 제대로 시작도 안했다고 경고했다'라며 '이제 전쟁 진행에 100% 만족한다'고 말했다. 푸틴 대통령은 이날 러시아 안전보장이사회 회의를 주재하고 크림대교 폭발의 배후로 우크라이나 정보 당국을 겨냥하면서 '이런 종류의 범죄에 대응을 하지 않는 것은 불가능하다'고 말했다. 그러나 드미트로 쿨레바 우크라이나 외무장관은 '러시아는 크림대교 사건 전에도 우크라이나에 계속 미사일 공격을 했다'며 '푸틴은 전투 패배로 절박한 상황이며 전황을 유리하게 바꾸려고 미사일 공포를 사용한다'고 반박했다. 우크라이나 구조대원들이 10일(현지시간) 수도 키이우 시내에서 러시아군의 미사일 공격을 당한 현장을 조사하고 있다. 이날 오전에 키이우 시내에 여러차례 폭발이 발생했다. 푸틴 대통령은 이번 공격이 국방부의 요청에 따라 이뤄졌다고 주장했는데, 이것이 사실이라면 이는 새로운 합동군 총사령관 세르게이 수로비킨의 첫번째 결정이라고 가디언은 전했다. 수로비킨 사령관과 함께 일했던 전 국방부 관계자는 가디언지에 '오늘 키이우에서 벌어진 일이 놀랍지 않았다. 그는 매우 무자비하고 사람 목숨을 신경 쓰지 않는다'며 '그의 손이 우크라이나인의 피로 뒤덮일까 걱정된다'고 말했다. 그러나 푸틴 대통령이 이날 공습으로 얻은 매파들의 호평은 오래가지 않을 수 있다. 그는 이번과 같은 대규모 미사일 공격은 러시아 영토 공격 시 대응으로 남겨둘 것이라고 말했는데 강경파들은 전면전을 원하고 있다. 정치 평론가인 세르게이 마르코프는 '러시아 여론은 대규모 공격과 우크라이나군이 사용할 가능성이 있는 인프라 완전파괴를 원한다'고 말했다. 한편으론 비판 세력의 목소리는 그다지 의미가 없을 수도 있어 보인다. 전쟁과 관련한 의사결정 과정은 여전히 불투명하다. 콜레스니코프 선임 연구원은 '푸틴 대통령으로선 매파와 극보수파의 불만에 대응하는 것이 중요하겠지만 나는 그들의 영향력을 과장하진 않을 것'이라며 '푸틴 자신이 가장 매파적이고 극보수적인 인물'이라고 말했다. 러시아에서 나오는 한 가지 이론은 푸틴 대통령이 악명 높은 새로운 군사령관을 임명함으로써 전쟁에서 국방부의 성과에 대한 분노를 줄이려고 한다는 것이다. 수로비킨 사령관과 2020년까지 함께 일했던 전 공군 중위인 글렙 이리소프는 '수로비킨은 강경파들을 선호하고 와그너 용병회사와도 좋은 관계를 유지한다'며 '그러나 그가 매우 잔인한 동시에 유능한 사령관이지만 모든 문제를 풀 수는 없을 것'이라고 말했다."
#text = "우크라이나 전쟁이 피의 보복 양상을 보이면서 러시아의 핵무기 사용 가능성에 대한 우려가 고조되자 서방이 푸틴 달래기에 나섰다. 조 바이든 미국 대통령은 블라디미르 푸틴 러시아 대통령을 이성적 행위자로 칭하고 정상회담 가능성도 열어놨다.바이든 대통령은 11일(현지시간) CNN 인터뷰에서 러시아가 핵무기를 사용할 것으로 보지 않는다고 밝혔다. 그는 푸틴 대통령이 전술 핵무기를 사용하는 게 얼마나 현실적이냐는 질문에 나는 그가 그럴 것으로 생각하지 않는다며 세계에서 가장 큰 핵보유국 중 하나의 세계적 지도자가 우크라이나에서 전략적 핵무기를 쓸 수도 있다고 말하는 건 무책임하다고 덧붙였다. 바이든 대통령은 지난 6일엔 아마겟돈(인류 최후 대전쟁)이 올 수 있다며 러시아의 핵무기 사용에 대한 우려를 내비쳤다.바이든 대통령은 푸틴을 이성적 행위자로 보는지에 관한 질문에 나는 그가 상당히 오판을 한 이성적 행위자라고 생각한다며 그는 키이우에서 환영받으리라 생각했을 텐데 완전 잘못 계산한 것이라고 강조했다.바이든 대통령은 다음 달 인도네시아에서 열리는 G20 정상회의에서 푸틴 대통령과의 회동에 여지를 남겼다. 그는 (회담 여부는) 그가 세부적으로 무슨 주제로 대화할지에 달려 있을 것이다면서 만일 푸틴 대통령이 러시아에서 투옥된 미국 농구 스타 브리트니 그리너에 관해 얘기하고자 한다면 대화는 열려 있다고 말했다.세르게이 라브로프 러시아 외무장관은 이날 러시아 국영TV에 러시아는 푸틴 대통령과 바이든 대통령 간 만남을 거절하지 않을 것이다. (미·러 정상회담을) 제안받으면 검토하겠다고 밝혔다.북대서양조약기구(NATO·나토)는 러시아의 핵 무기 사용 가능성에 촉각을 곤두세우면서도 아직까지 핵무기 사용 징후는 나타나지 않았다고 밝혔다. 옌스 스톨텐베르그 나토 사무총장은 벨기에 브뤼셀 본부 기자회견에서 러시아의 핵전력을 감시 중인데 태세에 변화는 없다고 말했다. 나토는 13일 핵전략 회의를 주재하고 내주 정례적인 핵 억지 훈련을 실시할 계획이다.영국의 정보기관도 러시아가 전술핵무기를 실제 사용하려면 아직 멀었다고 전망했다. 영국 일간 더타임스에 따르면 영국 도·감청 전문 정보기관 정보통신본부(GCHQ)의 제레미 플레밍 국장은 영국 싱크탱크인 왕립합동군사연구소(RUSI) 연설에서 어떤 기술적 준비 조치에도 관여하고 있다는 징후는 없다고 강조했다.러시아는 다만 당분간 우크라이나에 대한 공격 수위를 높일 것으로 보인다. 이에 볼로디미르 젤렌스키 우크라이나 대통령은 G7 정상과의 화상회담에서 현대적이고 효과적인 방공시스템을 확보하면 러시아 테러의 핵심인 로켓 공격도 중단될 것이라며 지원을 요청했다. 미국은 백악관 방공에 사용되고 있는 첨단지대공미사일체계(NASAMS) 2기를 곧 지원한다는 방침이다"
# text = "키릴로 티모셴코 대통령실 차장은 텔레그램을 통해 중대 기반 시설들에 자폭 드론을 동원한 또 다른 공격이 있었다고 말했다. 그는 공격을 받은 기반 시설이 어디인지는 언급하지 않았다.우크라이나 당국에 따르면, 우크라이나에서는 최근 몇주째 이란제 샤헤드-136 드론을 이용한 러시아군의 공격이 이어지고 있다. 이란은 러시아에 자국산 드론을 공급했다는 것을 부인하고 있다."
kw_extractor = yake.KeywordExtractor(lan='ko', top=50, COpy=True)
#DataCore(text=text, stopword_set=[], windowsSize=1, n=3)
#keywords = kw_extractor.extract_keywords(text)

keywords=pke_u_i9.extract_keywords(text)

for kw in keywords[:50]:
 	print(kw)

# DataCore 전처리 최종결과 실험
# a = DataCore(text=text, stopword_set=[], windowsSize=1, n=3)
# print(type(a))

# Datacore의 _build 전처리 실험 
# from segtok.segmenter import split_multi
# from segtok.tokenizer import web_tokenizer, split_contractions
# sentences_str = [ [w for w in split_contractions(web_tokenizer(s)) if not (w.startswith("'") and len(w) > 1) and len(w) > 0] for s in list(split_multi(text)) if len(s.strip()) > 0]
# print(sentences_str)

# pke에서의 raw text 전처리
# import sys
# sys.path.append(r'C:/Users/BIG-LGY/Documents/GitHub/pke_u')
# from readers import RawTextReader
# from data_structures import Document
# doc = Document()

# language = 'ko'
# parser = RawTextReader(language='ko') # 
# doc = parser.read(text=text, language='ko')
# print('doc.sentences 출력',[print(i.stems, end = ' ') for i in doc.sentences]) 
# # 한국어 pke에서의 특수한 stopword 리스트처리
# sentences = doc.sentences
# if language == 'ko':
#     stoplist = []
#     for sentence in sentences:
#         for idx, pos in enumerate(sentence.pos):
#             # if pos not in ('NNG','NNP') and pos != 'VV':
#             if pos not in ('NOUN','ADJ','VERB'):
#                 stoplist.append(sentence.stems[idx])
#     stoplist = list(set(stoplist))
#     print('self.stoplist 실행 :', stoplist)

# 한국어 pke에서의 특수한 stemming : ( 추후 결과가 다를시 코드 추가하여 변경되는지 여부 check 필요 )

#pke_u의 간단한 전처리 과정과 stopword 생성하는 코드 간소화 작업 
# from yake.korea_token import RawTextReader
# parser = RawTextReader() # 
# doc, stopword_korea = parser.read(text=text)
# print('doc.sentences 출력')
# print([print(i.stems, end = ' ') for i in doc[0].sentences]) 
# print(stopword_korea)

print('finish')