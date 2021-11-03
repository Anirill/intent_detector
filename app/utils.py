import json
import pymorphy2
from nltk.tokenize import wordpunct_tokenize


# WORD_DICT = 'word_dict_with_norm.json'
WORD_DICT = 'word_dict_with_norm.json'
REVERSED_LABELS = 'reversed_labels.json'
ALLOWED_CHARS = '0123456789?abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюяё'
DROPOUT = 0.1

morph = pymorphy2.MorphAnalyzer()


with open(WORD_DICT, 'r') as fp:
    word_dict = json.load(fp)

assert type(word_dict) == dict
assert word_dict['PAD'] == 1
assert word_dict['UNK'] == 0


with open(REVERSED_LABELS, 'r') as fp:
    reversed_labels = json.load(fp)
reversed_labels = {int(key): value for (key, value) in reversed_labels.items()}
label_num = {value: key for (key, value) in reversed_labels.items()}

assert len(reversed_labels) == 54
assert len(reversed_labels) == len(label_num)
assert label_num[reversed_labels[42]] == 42


with open("fill.txt", "r", encoding='utf-8') as file:
    lines = file.readlines()
for line in lines:
    line = line[:-2]
assert lines[0] == '- это что единственная шутка которую ты знаешь?\n'


def is_digit(word):
    try:
        _ = int(word)
    except:
        return False
    return True


def preprocess(line):
    line = line.lower()
    line = list(line)
    new_line = []
    for c in line:
        if c not in ALLOWED_CHARS:
            new_line.append(' ')
        elif c == '?':
            new_line.append(' QM ')
        else:
            new_line.append(c)
    result = ['PAD']
    words = (''.join(new_line)).split()
    for word in words:
        if is_digit(word):
            result.append('DIGIT')
        elif word == 'QM':
            result.append(word)
        else:
            result.append(morph.parse(word)[0].normal_form)
            result.append(word)
    return ' '.join(result)


def eval_preprocess(sentence, word_dict=word_dict):
    sent = preprocess(sentence)
    sent = wordpunct_tokenize(sent)
    result = []
    for word in sent:
        # print(word)
        if word in word_dict.keys():
            result.append(word_dict[word])
        else:
            result.append(word_dict['UNK'])
    return result
