#QR作成に必要なモジュール
import os
import pyqrcode
from pyzbar.pyzbar import decode
from PIL import Image
from pandas import DataFrame
import re

def make_qr():
    #QRに持たせる情報と画像ファイルの名前と大きさを指定する
    #scale >=2じゃないとcontentが読み込めない

    #10つのQRを作る
    qr_list = []
    for i in range(1,11):
        qr_list.append(pyqrcode.create(content = '{}'.format(i)))
        #file = はqrcodeを配置したいディレクトリと画像ファイルの名前をお好きなように
        print(qr_list)
        qr_list[i-1].png(file = 'qr/qr_storage/qr_{}.png'.format(i),scale = 2, module_color=[10,10,10,30])

def read_qr(number):
    #multiqrcodes decoder 
    data = DataFrame(decode(Image.open('qr/qr_storage/qr_{}.png'.format(number))))
    #問題番号だけ
    return data['data']

# print(make_qr())
qr_num = read_qr(1)
print(qr_num[0])
print(type(qr_num))
print(qr_num.values[0] == '1')
print(type(qr_num.values))
print(qr_num.values.astype('i') == 1)
