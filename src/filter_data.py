# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np


def conver_abbr(str):
    letter = re.sub("(\sa\s)|(^a\s)", " anh ".encode('utf-8'), str)
    letter = re.sub("(\se\s)|(^e\s)", " em ".encode('utf-8'), letter)
    letter = re.sub("(\sny\s)|(^ny\s)", " người_yêu".decode('utf-8'), letter)
    letter = re.sub("(\sy\s)|(^y\s)", " yêu ".decode('utf-8'), letter)
    letter = re.sub("(\siu\s)|(^iu\s)", " yêu ".decode('utf-8'), letter)
    letter = re.sub("(\sngta\s)|(^ngta\s)", " người_ta ".decode('utf-8'), letter)
    letter = re.sub("(\sa\s)|(^a\s)", " anh ".decode('utf-8'), letter)
    letter = re.sub("(\sng ta\s)|(^ng ta\s)", " người_ta".decode('utf-8'), letter)
    letter = re.sub("(\sntn\s)|(^ntn\s)", " như_thế_nào ".decode('utf-8'), letter)
    letter = re.sub("(\suh\s)|(^uh\s)", " ừ ".decode('utf-8'), letter)
    letter = re.sub("(\suhm\s)|(^uhm\s)", " ừ ".decode('utf-8'), letter)
    letter = re.sub("(\sdt\s)|(^dt\s)", " điện_thoại ".decode('utf-8'), letter)
    letter = re.sub("(\sđt\s)|(^đt\s)", " điện_thoại ".decode('utf-8'), letter)
    letter = re.sub("(\sm\s)|(^m\s)", " mình ".decode('utf-8'), letter)
    letter = re.sub("(\sko\s)|(^ko\s)", " không ".decode('utf-8'), letter)
    letter = re.sub("(\sk\s)|(^k\s)", " không ".decode('utf-8'), letter)
    letter = re.sub("(\skhong\s)|(^khong\s)", " không ".decode('utf-8'), letter)
    letter = re.sub("(\skh\s)|(^kh\s)", " không ".decode('utf-8'), letter)
    letter = re.sub("(\st\s)|(^t\s)", " tôi ".decode('utf-8'), letter)
    letter = re.sub("(\sbb\s)|(^bb\s)", " blackberry".decode('utf-8'), letter)
    letter = re.sub("(\sbb\d\s)|(^bb\d\s)", " blackberry ".decode('utf-8'), letter)
    letter = re.sub("(\s\dtr\s)|(^\dtr\s)", " num triệu ".decode('utf-8'), letter)
    letter = re.sub("(\ssam sung\s)|(^sam sung\s)", " samsung ".decode('utf-8'), letter)
    letter = re.sub("(\sss\s)|(^ss\s)", " samsung ".decode('utf-8'), letter)
    letter = re.sub("(\ssamsungsamsung\s)|(^samsungsamsung\s)", " samsung".decode('utf-8'), letter)
    letter = re.sub("(\stui\s)|(^tui\s)", " tôi ".decode('utf-8'), letter)
    letter = re.sub("(\sthui\s)|(^thui\s)", " thôi ".decode('utf-8'), letter)
    letter = re.sub("(\sbuon\s)|(^buon\s)", " buồn ".decode('utf-8'), letter)
    letter = re.sub("(\sbùn\s)|(^bùn\s)", " buồn ".decode('utf-8'), letter)
    letter = re.sub("(\shai(z+)\s)|(^hai(z+)\s)", " hãi ".decode('utf-8'), letter)
    letter = re.sub("(\sha(z+)\s)|(^ha(z+)\s)", " hãi ".decode('utf-8'), letter)
    letter = re.sub("(\sip\s)|(^ip\s)", " iphone ".decode('utf-8'), letter)
    letter = re.sub("(\siphone\d\s)|(^iphone\d\s)", " iphone ".decode('utf-8'), letter)
    letter = re.sub("(\svc\s)|(^vc\s)", " vãi_cứt ".decode('utf-8'), letter)
    letter = re.sub("(\stv\s)|(^tv\s)", " ti_vi ".decode('utf-8'), letter)
    letter = re.sub("(\sfan\s)|(^fan\s)", " phan ".decode('utf-8'), letter)
    letter = re.sub("(\slag\s)|(^lag\s)", " lác ".decode('utf-8'), letter)
    letter = re.sub("(\ssr\s)|(^sr\s)", " xin_lỗi ".decode('utf-8'), letter)
    letter = re.sub("(\scty\s)|(^cty\s)", " công_ty ".decode('utf-8'), letter)
    letter = re.sub("(\swá\s)|(^wá\s)", " quá ".decode('utf-8'), letter)
    letter = re.sub("(\sc\s)|(^c\s)", " cậu ".decode('utf-8'), letter)
    letter = re.sub("\s+", " ", letter)
    #print letter
    return letter


def preprocess(raw_comment):
    """
    Clearning and processing data
    :param raw_comment: a raw data, it contains some special charactor, upper case
    :return:
    """
    letter = re.sub("☺|😁|👍|😤|👅|👏|❤|😍|💲|🤔|😉|💪|😈|👿|👽|😌|😋|😅|😅|😂", " ", raw_comment, flags=re.UNICODE)
    letter = \
        re.sub("\\.|,|/|-|\\?|<|>|/|:|;|'|\\[|\\]|\\{|\\}|\\\\|\\||=|\\+|-|\\(|\\)|\\*|&|\\^|%|$|#|@|!|~|`",
               " ", letter, flags=re.UNICODE)
    letter = re.sub('“|”|"', " ", letter)
    letter = re.sub("(^\d+\s)|(\s\d+\s)", " num ", letter)
    letter = re.sub("(^\d+\s)|(\s\d+\s)", " num ", letter)
    letter = re.sub("\s\w\s", " ", letter)
    letter = re.sub("\s+", " ", letter)
    letter = letter.lower()
    letter = conver_abbr(letter)
    return letter


def read_file_text(filename):
    data = pd.read_csv(filename, sep="\n", header=None, names=['content'], encoding='utf-8')
    return data

def expadding_abbr_file(filename):
    data = read_file_text(filename)
    with open("../data/smooth_out1_3.txt", "w") as text_file:
        for l in data["content"]:
            ex_abbr = l.lower()
            ex_abbr = conver_abbr(ex_abbr)
            s = ex_abbr.encode('utf-8')
            text_file.write(s)
            text_file.write("\n")

def write_prepocess_to_file(filename):
    data = read_file_text(filename)
    smooth_data = pd.DataFrame()
    smooth_data_pro = []
    for s in data["content"]:
        s_pre = preprocess(s).strip()
        if s_pre and ~s_pre.isspace():
            smooth_data_pro.append(s_pre.encode('utf-8'))
    file_out = "smooth" + filename
    smooth = pd.DataFrame(data={"content": smooth_data_pro})
    smooth.to_csv("../data/smooth/smooth_out1_3.csv", index=False, quoting=3, encoding='utf-8')


if __name__ == '__main__':
    option = "preprocess"
    if option == "exp":
        expadding_abbr_file("../data/raw/SA-training_negative.txt")
    else:
        write_prepocess_to_file("../data/token/smooth_out1_3.txt")

