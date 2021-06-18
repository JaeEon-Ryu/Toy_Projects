#https://github.com/lumyjuwon/KoreaNewsCrawler
#training data
from korea_news_crawler.articlecrawler import ArticleCrawler

Crawler = ArticleCrawler()
Crawler.set_category("정치", "경제", "사회","생활문화","세계","IT과학")
Crawler.set_date_range(2019, 12, 2019, 12)
Crawler.start()

###############################################################
# 코드 통합
import csv
import os

os.chdir("C:\\Users\\Yoo Jae Un\\데이터사이언스_실습\\Csv") // C

category = ['IT과학', '경제', '사회', '생활문화', '세계', '정치']

body_unity = open('body_unity.csv', 'w', encoding='UTF-8')
wcsv = csv.writer(body_unity)

count = 0

for category_element in category:
    file = open('Article_' + category_element + '.csv', 'r', encoding='UTF-8', newline="")
    line = csv.reader(file)
    try:
        for line_text in line:
            wcsv.writerow([line_text[1], line_text[4])
    except:
        pass

#############################################################

# 코드 셔플
import csv
import random
import os

os.chdir("C:\\Users\\Yoo Jae Un\\데이터사이언스_실습\\")  # Csv가 있는 경로 설정

file = open('test_data.csv', 'r', encoding='euc-kr')
line = file.readlines()
random.shuffle(line)
rcsv = csv.reader(line)

body_write = open('test_data_shuffled.csv', 'w', encoding='euc-kr', newline="")
wcsv = csv.writer(body_write)

for i in rcsv:
    try:
        wcsv.writerow([i[1].strip(), i[4]])
    except:
        pass
print('완료')

#################################################################
# 형태소 분석 및 word to vector
from konlpy.tag import Twitter
from gensim.models import Word2Vec
import csv

twitter = Twitter()

file = open("body_shuffled.csv", 'r', encoding='euc-kr')
line = csv.reader(file)
token = []
embeddingmodel = []

category = ('정치', '경제', '사회', '생활문화', '세계', '과학')
tttt = 1

for i in line:
    print(tttt)
    sentence = twitter.pos(i[1], norm=True, stem=True)
    temp = []
    temp_embedding = []
    all_temp = []
    for k in range(len(sentence)):
        print(sentence[k][0])
        temp_embedding.append(sentence[k][0])
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    all_temp.append(temp)
    embeddingmodel.append(temp_embedding)

    category_number_dic = {'정치': 0, '경제': 1, '사회': 2, '생활문화': 3, '세계': 4, 'IT과학': 5}
    all_temp.append(category_number_dic.get(category))
    token.append(all_temp)
    tttt += 1
print("토큰 처리 완료")

embeddingmodel = []
for i in range(len(token)):
    temp_embeddingmodel = []
    for k in range(len(token[i][0])):
        temp_embeddingmodel.append(token[i][0][k])
    embeddingmodel.append(temp_embeddingmodel)
# max_vocab size 10000000 개당 1 GB 메모리 차지
embedding = Word2Vec(embeddingmodel, size=300, window=5, min_count=10, iter=5, sg=1, max_vocab_size=360000000)
embedding.save('post.embedding')
