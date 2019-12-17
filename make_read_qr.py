#!/usr/bin/env python
# coding: utf-8
#QR作成に必要なモジュール
import os
import pyqrcode
from pyzbar.pyzbar import decode
from PIL import Image
from pandas import DataFrame
import re
import hashlib

"""
@ hashword: ハッシュ関数の引数
試験セット各々を憶えて管理するために使用
"""
def make_qr(num=20, hash=False, hashword=None):

	"""
	'input word' -> hash() -> QR codeは同じinputからは一意に決まる
	QRを読み取ればhash値も求まるため、用紙の問題識別用QRで管理（将来時間があれば）
	"""
	if hash==True:
		hs = hashlib.md5(hashword.encode()).hexdigest()
		hashqr = pyqrcode.create(content = 'hs')
		hashqr.png(file = 'qr/qr_storage/hash_{}.png'.format(hs), scale = 2, module_color=[0,0,0,0])
		return 'qr/qr_storage/hash_{}.png'.format(hs)

	qr_list = []
	for i in range(1,num+1):
		#QRに持たせる情報と画像ファイルの名前と大きさを指定する
		#scale >=2じゃないとcontentが読み込めない
		qr_list.append(pyqrcode.create(content = '{}'.format(i)))
		qr_list[i-1].png(file = 'qr/qr_storage/qr_{}.png'.format(i),scale = 2, module_color=[0,0,0,0])

def read_qr(number):
    #multiqrcodes decoder 
    data = DataFrame(decode(Image.open('qr/qr_storage/qr_{}.png'.format(number))))
    #問題番号だけ
    return data['data']

make_qr(20, hash=True, hashword="python")
