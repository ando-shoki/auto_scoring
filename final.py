import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
# from keras.preprocessing import image
import keras.preprocessing.image
#from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
from keras_preprocessing import image

#model = load_model('auto_scoring/model.h5')#学習済みモデルをロードする
model = tf.keras.models.load_model('model_mnist.h5')#学習済みモデルをロードする

app = Flask(__name__)

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

image_size = 28

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

answer = ["0","1","2","3","4","5","6","7","8","9"]

@app.route('/',methods = ['GET','POST'])
def upload_file():
    #bugが出たらkerasがやばいかも
    #global graph
    #with graph.as_default():
    if request.method == 'POST':
        '''
        @img_file : 回答用紙のファイル
        '''
        if ('img_file' not in request.files) :
            flash('ファイルがありません')#flashはユーザーに挙動が正しいかを知らせる
            return redirect(request.url)
        img_file = request.files['img_file']
        
        if img_file.filename == '' :
            flash('ファイルがありません')
            return redirect(request.url)
        if img_file and allowed_file(img_file.filename):
            
            img_filename = secure_filename(img_file.filename)
            img_file.save(os.path.join(UPLOAD_FOLDER,img_filename))
            img_filepath = os.path.join(UPLOAD_FOLDER, img_filename)
            
            # qr_img = keras.preprocessing.image.load_img(img_filepath ,grayscale=True,target_size=(image_size,image_size))
            qr_imgcv = cv2.imread(img_filepath)
            # gray_img = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
            gray_imgcv = cv2.cvtColor(qr_imgcv, cv2.COLOR_BGR2GRAY)
            blurred_img = cv2.GaussianBlur(gray_imgcv, (11,11),0)
            barcodes = pyzbar.decode(blurred_img)
            
    
            #the picture's number
            i=1

            ans_correct  = 0
            qr_list = np.empty((1,3))
            for  barcode in barcodes:
                (x, y, w, h) = barcode.rect
                #以下imwriteは確認
                #binary image 
                barcodeData = barcode.data.decode("utf-8")
                crop_cut = gray_imgcv[y:y+h ,x+310 :x+w+310]
                th, binary = cv2.threshold(crop_cut,125, 255, cv2.THRESH_BINARY)
                #cv2.imwrite('crop_cut_binary_{}.jpg'.format(i),binary)

                #center the number and padding 
                binary2black = 255 - binary              
                #cv2.imwrite('binary2black_{}.jpg'.format(i), binary2black)

                
                _, thresh = cv2.threshold(binary, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, hierarchy = cv2.findContours(thresh, 3, cv2.CHAIN_APPROX_NONE)
                cnt = contours[0]
                X, Y, W, H = cv2.boundingRect(cnt) 
                rectangle = binary2black[Y:Y+H,X:X+W]
                Padding = cv2.copyMakeBorder(rectangle,50,50,80,80,cv2.BORDER_CONSTANT,value=[0,0,0])
                #cv2.imwrite('Padding_{}.jpg'.format(i),Padding)

                #Resize and sharpen image
                resized = cv2.resize(Padding,(28,28),interpolation=cv2.INTER_AREA)     
                #cv2.imwrite('resized_{}.jpg'.format(i), resized)

                #increase the picture's number
                i += 1

                #Predict and show the result
                pred = model.predict(resized.reshape(1, 28, 28)).argmax()            
                print("No.{0} question's answer is {1}".format(barcodeData,pred))

                qr_que = int(barcodeData)
                qr_ans = int(pred)
                true_ans = int(answer[qr_que])
                    
                if qr_ans == true_ans:
                    print('The answer is correct')
                    ans_correct += 1
                else:
                    print('The answer is incorrect')
                qr_list = np.append(qr_list, [[qr_que, qr_ans, true_ans]], axis=0)
        
        qr_list = np.delete(qr_list, 0, 0)
        qr_list = qr_list[qr_list[:,0].argsort(), :]
        x1 = (ans_correct / 10.0) * 100
        print('If the prediction is 100% correct. The correct rate of answer is {} %'.format(x1))
        return render_template('result.html', res_list=qr_list)                
    #img_read = cv2.imread('test2.jpg')
    #read_pred(img_read)
    return render_template('index.html', answer = '')

#直接実行した時のみに動く
if __name__ == '__main__':
    app.run(port=8000, debug=True)




