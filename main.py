import pandas as pd
import os.path
import re
import olefile
import zlib
import struct
from collections import defaultdict
import pickle
import copy
from kiwipiepy import Kiwi
import kiwipiepy_model
kiwi = Kiwi(model_type='sbg')  ##

def get_hwp_text(filepath):
    with olefile.OleFileIO(filepath) as f:
        dirs = f.listdir()

        # HWP 파일 검증
        if ["FileHeader"] not in dirs or \
                ["\x05HwpSummaryInformation"] not in dirs:
            raise Exception("Not Valid HWP.")

        # 문서 포맷 압축 여부 확인
        header = f.openstream("FileHeader")
        header_data = header.read()
        is_compressed = (header_data[36] & 1) == 1

        # Body Sections 불러오기
        nums = []
        for d in dirs:
            if d[0] == "BodyText":
                nums.append(int(d[1][len("Section"):]))
        sections = ["BodyText/Section" + str(x) for x in sorted(nums)]

        # 전체 text 추출
        text = ""
        for section in sections:
            bodytext = f.openstream(section)
            data = bodytext.read()
            if is_compressed:
                unpacked_data = zlib.decompress(data, -15)
            else:
                unpacked_data = data

            # 각 Section 내 text 추출
            section_text = ""
            i = 0
            size = len(unpacked_data)
            while i < size:
                header = struct.unpack_from("<I", unpacked_data, i)[0]
                rec_type = header & 0x3ff
                rec_len = (header >> 20) & 0xfff

                if rec_type in [67]:
                    rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                    section_text += rec_data.decode('utf-16', errors='ignore')
                    section_text += "\n"

                i += 4 + rec_len

            text += section_text
            text += "\n"

        # post-processing noise included text
        noise_included = text[16:]  # [0:16]은 쓸모 없는 내용
        pp_text = re.sub("[^가-힣A-Za-z0-9 ]{1}x{1}[^가-힣]*", "", noise_included)  # \x00 과 같은 단어들 일괄 제거
        pped_text = re.sub("[^가-힣A-Za-z0-9 -=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\n]", "", pp_text)  # 한글, 영어, 특수문자 제외 하구 모두 제거

        # good by nogada (be going to delete code below ...)
        # pped_text = re.sub("[一-龥]", "", pp_text)
        # pp_text = re.sub("\x0b漠杳\x00\x00\x00\x00\x0bs", "", pp_text)
        # pp_text = re.sub("\x15湯湷\x00\x00\x00\x00\x15", "", pp_text)
        # pp_text = re.sub("\x15湰灧\x00\x00\x00\x00\x15", "", pp_text)
        # pp_text = re.sub("\x0b漠杳\x00\x00\x00\x00\x0b", "", pp_text)
        # pp_text = re.sub("\x10慤桥\x00\x00\x00\x00\x10", "", pp_text)
        # pp_text = re.sub("\x10慤桥\x00\x00\x00\x00\x10", "", pp_text)
        # pp_text = re.sub("\x15桤灧\x00\x00\x00\x00\x15", "", pp_text)
        # pped_text = re.sub("\x12湯慴\x00\x00\x00\x00\x12", "", pp_text)

        return pped_text


def preprocessor(raw_text):
    '''
    :param raw_text: raw text from (1)pre-processed hwp file or (2) txt file
    :return: list of sentences
    '''
    # 괄호 안과 밖으로 나누어 텍스트 처리
    # 괄호 안 텍스트
    inparen_text = re.findall('\(.*?\)', raw_text)  # non-greedy == .*?
    inparen_text = [x[1:-1] for x in inparen_text]  # 괄호제거
    inparen_sents_list = list(set(sum([re.split(',', x) for x in inparen_text], [])))
    inparen_sents_list = [x.strip() for x in inparen_sents_list]

    # 괄호 밖 텍스트
    outparen_text = re.sub('\(.*?\)', '', raw_text)  # 괄호 안은 삭제
    outparen_text = re.sub('\x07', '', outparen_text)
    outparen_sents_list = re.split('\n', outparen_text)  # 줄바꿈 단위로 나누기
    outparen_sents_list = [x.strip() for x in outparen_sents_list]
    outparen_sents_list = [x for x in outparen_sents_list if x != '']  # 공백 문자 제거

    # 괄호 안 + 괄호 밖
    splited_sents = inparen_sents_list + outparen_sents_list

    return splited_sents

def matcher(ngram, template):
    # re.match와 혼동 가능하기 때문에 matcher로 변경
    if len(ngram) != len(template):
        # print('not same length with', ngram, r'/', template)
        return False
    for n, t in zip(ngram, template):
        if not (t in n):
            return False
            break
        else:
            continue
    return True



def tokenizer_kiwi(sent, pre_spacing=True):  # [(형태소1, 품사1), (형태소2, 품사2), ...] 형태로 결과를 리턴
    if pre_spacing:
        sent = kiwi.space(sent, reset_whitespace=True)
    result = list(map(lambda x: x[0] + "/" + x[1], kiwi.tokenize(sent)))

    return result

def templated_ngram_extractor(docs, pos_templates=list(), neg_templates=list(),
                              min_count=5, n_range=(1, 5)):
    '''
    :param docs: list of doc (e.g.) [doc1, doc2, ...]
    :param pos_templates: list of templates (e.g.) [tuple(['/NN']), ...] or [(/NN,), ...]
    :param neg_templates:
    :param min_count:
    :param n_range:
    :return:
    '''

    def to_ngrams(words_list, n):
        ngrams = []
        for b in range(0, len(words_list) - n + 1):
            ngrams.append(tuple(words_list[b:b + n]))
        return ngrams

    def find_matched_ngram(ngrams):
        matcheds = []
        for ngram in ngrams:

            if pos_templates:
                for pos_template in pos_templates:  # pos_templates 이 있는 경우
                    if matcher(ngram, pos_template):

                        if neg_templates:  # neg_templates 이 있는 경우
                            ox_tester = False  # 기본값 False
                            for neg_template in neg_templates:
                                if matcher(ngram, neg_template):
                                    ox_tester = False
                                    break
                                else:
                                    ox_tester = True
                                    continue  # 수정
                            if ox_tester:
                                matcheds.append(ngram)  # 모두 잘 통과하면 추가
                        else:
                            matcheds.append(ngram)

            else:
                if neg_templates:
                    ox_tester = False  # 기본값 False
                    for neg_template in neg_templates:
                        if matcher(ngram, neg_template):
                            ox_tester = False
                            break
                        else:
                            ox_tester = True
                    if ox_tester:
                        matcheds.append(ngram)

                else:
                    matcheds.append(ngram)

        return matcheds

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:
        tokenized_words_list = tokenizer_kiwi(doc)
        for n in range(n_begin, n_end + 1):
            ngrams = to_ngrams(tokenized_words_list, n)
            ngrams = find_matched_ngram(ngrams)
            for ngram in ngrams:
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram: count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter

def dict_merger(dict1, dict2):
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if key in dict1:
            result[key] = dict1[key] + value
        else:
            result[key] = value
    return result

# templates = [
#     ('/NNG',),
#     ('/VV',),
#     ('/VA',),
#     ('/NNG', '/NNG',),
#     ('/NNG', '/SO',),
#     ('/NNG', '/VCP',),
#     ('/NNG', '/XSA',),
#     ('/NNG', '/XSN',),
#     ('/NNG', '/XSV',),
#     ('/NP', '/NNG',),
#     ('/NNG', '/JC', '/NNG',),
#     ('/NNG', '/JC', '/NNP',),
#     ('/NNG', '/JKG', '/NNG',),
#     ('/NNG', '/JKO', '/NNG',),
#     ('/NNG', '/JKO', '/VV-R',),
#     ('/NNG', '/JKO', '/VV',),
#     ('/NNG', '/NNG', '/NNG',),
#     ('/NNG', '/NNG', '/XSN',),
#     ('/NNG', '/NNG', '/NNG', '/NNG',)
# ]


print("한글 파일을 불러올 폴더 주소*를 입력하세요. (예:C:\\Data\\예시파일)")
folder_path = input()
# folder_path = r"C:\Users\nuoguri\Desktop\UC Messenger Download files\test"
if os.path.exists(folder_path):  # 주소 확인
    print("입력 성공^^")
else:
    print("없는 주소 잖아요. 장난합니까?")

print("이제 파일을 추출합니다...")

file_path_list = list()
for (root, directories, files) in os.walk(folder_path):
    file_path_list.append([root, files])  # [루트, [파일명1, 파일명2 ...]] (list)을 추출



folder_path = file_path_list[0][0]
folder_name = re.sub(r'/.*/', '', folder_path)
k = 0
pickle_file_name_list = list()
pickle_data_list = list()
while k < len(file_path_list[0][1]):

    vocab_dict = dict()
    file_name = file_path_list[0][1][k]
    file_path = os.path.join(folder_path, file_name)
    if '.hwp' in file_name:
        print("한글 파일 맞네예:", file_name)
        pickle_file_name_list.append(file_name)
        raw_text = get_hwp_text(file_path)
        sents = preprocessor(raw_text)
        counted_ngram_5 = templated_ngram_extractor(sents,  # pos_templates=templates,
                                                    min_count=1,
                                                    n_range=(1, 4))
        vocab_dict = dict_merger(vocab_dict, counted_ngram_5)
        print('작업완료!',

              r'총',
              len(pickle_file_name_list),
              '개 한글 파일 중',
              k + 1,
              '번째 파일입니다.'
              "작업 파일명:",
              file_name)
        k += 1

        if vocab_dict:
            # Save with pickle
            pickle_data_list.append(vocab_dict)  # 각 파일별로 어휘 추출된 결과 붙여넣기

    else:
        print("한글 파일이 아닌 파일:", file_name)
        k += 1
        pass

print("만약 제대로 되었다면, 작업완료! 메세지가 진작 떴을 겁니다. 봤나요?")

# print("산출될 파일을 저장할 폴더 주소를 입력하세요.")

# 저장한 vocab pickle file 불러오기
# pickle_file_folder = folder_path + r'/vocab_extracted'
# pickle_file_path_list = list()
# for (root, directories, files) in os.walk(pickle_file_folder):
#     pickle_file_path_list.append([root, files])  # [루트, [파일명1, 파일명2 ...]] (list)을 추출

# pickle_file_name_list = pickle_file_path_list[0][1]
# pickle_folder_path = pickle_file_path_list[0][0]
# pickle_data_list = list()
# for n in range(len(pickle_file_path_list[0][1])):
#     pickle_file_path = os.path.join(pickle_folder_path, pickle_file_name_list[n])
#     with open(pickle_file_path, 'rb') as pf:
#         data_dict = pickle.load(pf)
#         pickle_data_list.append(data_dict)
#         print("완료!:", pickle_file_name_list[n])


def restore_word_from_pos(possed_token):
    #keyw == possed_token
    # restore word from tokenized results
    before_join=list()
    # possed_token=('나/NP',)
    for x in possed_token:
        pos_e_last = x.rfind(r'/')
        before_join.append(tuple([x[:pos_e_last], x[(pos_e_last+1):]]))
    # before_join = [tuple(x.split(r'/')) for x in possed_token]
    restored_word = kiwi.join(before_join)
    return restored_word

restore_word_from_pos(('나/NP','가/JX'))

def edit_pos_error(input_dict):
    tmp = dict()
    for x, freq in input_dict.items():
        if '//NNG' in str(x):
            pass
            # print('발견!!', str(x))
        elif '/가/NNG' in str(x):
            pass
            # print('발견!!', str(x))
        elif '/가외/NNG' in str(x):
            pass
            # print('발견!!', str(x))
        elif '//SP' in str(x):
            pass
            # print('발견!!', str(x))
        else:
            tmp[x] = freq
    return tmp

# pickle_data_list[0]
# {('시/EP', '었/EP', '습니다/EF', './SF'): 1, ... ('었/EP', '습니다/EF', './SF', '>/SSC'): 1}

new_pickle_data_list = list() # new_pickle_data_list의 구성요소들은 dict...
for x in pickle_data_list:
    new_pickle_data_list.append(edit_pos_error(x))




data_list=list()
for x in new_pickle_data_list:
    _ = list()
    for key, freq in x.items():
        # print(key) # key만 뽑아서 쓴다, freq는 말고
        # print(restore_word_from_pos(key))
        _.append((re.sub(" ", "", restore_word_from_pos(key)), key))  # (수정/복원된 형태의 어휘(한국의집), 형태소 나뉜 것(한국+의+집))
    data_list.append(_)  # 개별 파일들에서 추출한 어휘가 리스트 내의 리스트로 들어간다.


print("추출한 데이터를 정리하는 중.....")

data_dict_list=list()
for z in data_list:
    tmp=dict()
    for x,y in z:
        if x not in tmp:
            tmp[x]=[y]
        else:
            tmp[x].append(y)
    data_dict_list.append(tmp)


common_data=dict()  # 얘가 행 데이터
for each_data_dict in data_dict_list:
    for x, y in each_data_dict.items():
        if x not in common_data:
            common_data[x]=[y]
        else:
            common_data[x].append(y)

common_data_list=[x for x,y in common_data.items()]  #어휘들 전체 리스트
common_pos_data_list=[y for x,y in common_data.items()]  #어휘들 전체 리스트

# sum([[('N/SL',)], [('N/SL',)]], [])
def target_text_searcher(target, test):
    tmp = list()
    idx = -1
    while True:
        idx = test.find(target, idx + 1)
        if idx == -1:
            break

        from_n = idx - 10
        to_n = idx + 10
        if from_n < 0:
            from_n = 0
        if to_n > idx+len(target):
            to_n = idx+len(target)+3
        if to_n > len(test):
            to_n = len(test)
        tmp.append(test[from_n:idx]+'*'+test[idx:idx+len(target)]+'*'+test[idx+len(target):to_n])
    return tmp


rows=list()  # rows == pandas' rows...
for word, wpos_list in common_data.items():
    row = list()
    for data_set in data_dict_list:  #1
        if word in data_set: # 해당 데이터 내에 있는지 확인
            row.append(1)
        else:
            row.append(0)
    row.append(sum(row))  #2
    row.append(list(set(sum(wpos_list, []))))  #3
    # 해당 단어를 서칭하여   #4
    rows.append(row)

print("단어가 쓰인 문맥을 찾는중...(좀 오래 걸려요. 놀라지마셈...)")
from tqdm import tqdm
context_infos=list()
for k in tqdm(range(len(rows))):
    pos_words_list = common_pos_data_list[k]
    words_list = list(set([restore_word_from_pos(x[0]) for x in pos_words_list]))
    context_info = str()
    for m in range(len(rows[k][:-2])):
        if rows[k][:-2][m] ==1:
            file_name = pickle_file_name_list[m]
            file_path = os.path.join(folder_path, file_name)
            if '.hwp' in file_name:
                raw_text = get_hwp_text(file_path)
                infile_txt = preprocessor(raw_text)
                context_texts=list()
                for excerpt_text in infile_txt:
                    for word in words_list:
                        context_texts += target_text_searcher(word, excerpt_text)
                if context_texts:
                    context_info += str(file_name)+': '+str(context_texts)+'\n'
    context_infos.append(context_info)


import copy
def make_list_same_length(list, maxlen=4):
    return_list = copy.deepcopy(list)
    while len(return_list) < maxlen:
        return_list += ['']
    return return_list

rows_=list()
for x, y in zip(rows, context_infos):
    tmp = list()
    tmp = x[:-1] + [y]  #문맥
    tmp += [x[-1]]  # 형태소 분석 결과들의 리스트
#    if x[-1]:
    only_pos = [re.sub('.*/', '', x) for x in list(x[-1][0])]
    tmp += make_list_same_length(only_pos)  # 형태소 분석 결과 맨 첫 번째 꺼 기준 튜플을 리스트로 만들어서 더하기 # [('문항/NNG', '에/JKB', '는/JX', '배점/NNG')]
#    else:
#        tmp += ['','','','']
    rows_.append(tmp)

# for x, y in zip([1,2,3,4], [5,6,7,8]):
#     print(x+y)



result = pd.DataFrame(rows_, columns=pickle_file_name_list+['sum', '문맥정보', '형태소', '형1', '형2', '형3', '형4'], index=common_data_list)
result.sort_values(by = 'sum', ascending = False, inplace=True)

result.to_csv(os.path.join(folder_path, 'result.csv'))

print("파일이 생성되었습니다. 결과 파일을 확인하세요.")
