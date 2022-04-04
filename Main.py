import glob
import PIL
import numpy as np
import re
import cv2
import pytesseract
import PIL.Image
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton

class MainApp(MDApp):
    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        self.save_img_button = MDRaisedButton(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .1},
            size_hint=(None, None))
        self.save_img_button.bind(on_press=self.take_picture)
        layout.add_widget(self.save_img_button)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()
        # frame initialize
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def qr_detect(self, *args):
        qr = 1
        qrCodeDetector = cv2.QRCodeDetector()
        decodedText, points, _ = qrCodeDetector.detectAndDecode(self.image_frame)
        if points is not None:
            nrOfPoints = len(points[0])
            # print(nrOfPoints)
            # print(points)
            for i in range(nrOfPoints):
                nextPointIndex = (i + 1) % nrOfPoints
                cv2.line(self.image_frame, tuple((int(points[0][i][0]), int(points[0][i][1]))),
                         tuple((int(points[0][nextPointIndex][0]), int(points[0][nextPointIndex][1]))), (0, 255, 0), 3)
            print(decodedText)
            qr = 1
            print("QR - FOUND")
            qr_image = "picture_after_qr.png"
            cv2.imwrite(qr_image, self.image_frame)
        else:
            qr =-1
            print("QR - NOT FOUND")
            qr_image = "picture_after_qr.png"
            cv2.imwrite(qr_image, self.image_frame)
        return qr

    def logo_detect(self, *args):
        logo =1
        file_list = glob.glob('logos/*')
        flag = 0
        for i in file_list:
            img1 = cv2.imread(i, 0)
            MIN_MATCH_COUNT = 11
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(self.image_frame, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.76 * n.distance:
                    good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                try:
                    flag = 1
                    dst = cv2.perspectiveTransform(pts, M)
                    self.image_frame = cv2.polylines(self.image_frame, [np.int32(dst)], True, (255, 255, 0), 5, cv2.LINE_AA)
                    logo = 1
                    print("LOGO - FOUND")
                    logo_image="picture_after_logo.png"
                    cv2.imwrite(logo_image, self.image_frame)
                except:
                    flag = 1
                    logo = -1
                    print("LOGO - Error")
                    #return self.image_frame
        if (flag == 0):
            logo = -1
            print("LOGO - NOT FOUND")
            logo_image = "picture_after_logo.png"
            cv2.imwrite(logo_image, self.image_frame)
        return logo

    def date_detect(self, *args):
        date = 1
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        month_dict = dict(jan='01', feb='02', mar='03', apr='04', may='05', jun='06', jul='07', aug='08', sep='09',
                          oct='10', nov='11', dec='12')

        def word_to_num(string):
            s = string.lower()[:3]
            return month_dict[s]

        def date_converter(string):
            results = []
            day = '01'
            month = '01'
            year = '1900'

            date = re.search('(0?[1-9]|[12][0-9]|3[0-1])(\.|-|/)(0?[1-9]|1[0-2])(\.|-|/)(20[01][0-9]|\d\d)', string)

            date1 = re.search('(0?[1-9]|1[0-2])(\.|-|/)(0?[1-9]|[12][0-9]|3[0-1]|[00])(\.|-|/)(20[01][0-9]|\d\d)',
                              string)

            string = string.replace("'", ' ').replace("Jan", " Jan ").replace("JAN", " Jan ").replace("Feb",
                                                                                                      " Feb ").replace(
                "FEB", " Feb ").replace("Mar", " Mar ").replace("MAR", " Mar ").replace("Apr", " Apr ").replace("APR",
                                                                                                                " Apr ").replace(
                "May", " May ").replace("MAY", " May ").replace("Jun", " Jun ").replace("JUN", " Jun ").replace("Jul",
                                                                                                                " Jul ").replace(
                "JUL", " Jul ").replace("Aug", " Aug ").replace("AUG", " Aug ").replace("Sep", " Sep ").replace("SEP",
                                                                                                                " Sep ").replace(
                "Oct", " Oct ").replace("OCT", " Oct ").replace("Nov", " Nov ").replace("NOV", " Nov ").replace("Dec",
                                                                                                                " Dec ").replace(
                "DEC", " Dec ")

            month1 = re.search(
                '(0?[1-9]|[12][0-9]|3[0-1])(?:st|nd|rd|th)?\s*[-|/|.\s]\s*(Jan(?:uary)?|JAN(?:UARY)?|Feb(?:ruary)?|FEB(?:RUARY)?|Mar(?:ch)'
                '?|MAR(?:CH)?|Apr(?:il)?|APR(?:IL)?|May|MAY|June?|JUNE?|July?|JULY?|Aug(?:ust)?|AUG(?:UST)?|Sept(?:ember)?|SEPT'
                '(?:EMBER)?|Sep(?:tember)?|SEP(?:TEMBER)?|Oct(?:ober)?|OCT(?:OBER)?|Nov(?:ember)?|NOV(?:EMBER)?|Dec(?:ember)?|DEC(?:EMB'
                'ER)?).?\s*[-|/|.\s]\s*(20[01][0-9]|\d\d)', string)

            month2 = re.search(
                '(Jan(?:uary)?|JAN(?:UARY)?|Feb(?:ruary)?|FEB(?:RUARY)?|Mar(?:ch)?|MAR(?:CH)?|Apr(?:il)?|APR(?:IL)?|May|June?|JUNE?|'
                'July?|JULY?|Aug(?:ust)?|AUG(?:UST)?|Sept(?:ember)?|SEPT(?:EMBER)?|Sep(?:tember)?|SEP(?:TEMBER)?|Oct(?:ober)?|OCT(?:OBER)?|Nov(?:ember)?|NOV(?:EM'
                'BER)?|Dec(?:ember)?|DEC(?:EMBER)?).?\s*[-|/|.\s]\s*(0?[1-9]|[12][0-9]|3[0-1])(?:st|nd|rd|th)?\s*[-|/|.,\s]\s*(20[01][0-9]|\d\d)'
                , string)

            if date:
                day = date.group(1)
                month = date.group(3)
                year = date.group(5)
                start = date.start()
                end = date.end()
            elif date1:
                day = date1.group(3)
                month = date1.group(1)
                year = date1.group(5)
                start = date1.start()
                end = date1.end()
            elif month1:
                day = month1.group(1)
                month = word_to_num(month1.group(2))
                year = month1.group(3)
                start = month1.start()
                end = month1.end()
            elif month2:
                day = month2.group(2)
                month = word_to_num(month2.group(1))
                year = month2.group(3)
                start = month2.start()
                end = month2.end()
            else:
                return ["Not Found", 0, 0]

            month = month.zfill(2)
            day = day.zfill(2)
            if day == '00':
                day = '01'
            if year is not None and len(year) == 2:
                year = '20' + year
            results.append(day + "-" + month + "-" + year)
            return [results, start, end]

        def combine_words(df):
            string = ""
            for i in range(len(df)):
                string = string + str(df.loc[i, 'text'])
            print(string)
            return string

        # image = "4.jpeg"
        # img = cv2.imread(image)
        #image_final = cv.imread("picture_after_logo.png")
        #cv2.imwrite("ddd",)
        img = cv2.resize(self.image_frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img = cv2.blur(img, (2, 2))
        img = cv2.GaussianBlur(img, (1, 1), 0)

        flag = 0
        data = pytesseract.image_to_data(img, output_type='data.frame')
        data = data.dropna(axis=0, subset=['text'])
        data = data.reset_index()
        text = combine_words(data)
        D = date_converter(text)
        print(D[0])
        if D[0] == "Not Found":
            flag = 1
            data1 = pytesseract.image_to_data(PIL.Image.open("picture_after_logo.png").convert("RGB"), lang='eng', config='--psm 6',
                                              output_type='data.frame')
            data1 = data1.dropna(axis=0, subset=['text'])
            data1 = data1.reset_index()
            text1 = combine_words(data1)
            D = date_converter(text1)
            print(D[0])
            if D[0] == "Not Found":
                flag = 2
                date = -1
                im = PIL.Image.open("picture_after_logo.png")
                dimg = ImageOps.grayscale(im)
                contrast = ImageEnhance.Contrast(dimg)
                eimg = contrast.enhance(5.5)
                data2 = pytesseract.image_to_data(eimg, output_type='data.frame')
                data2 = data2.dropna(axis=0, subset=['text'])
                data2 = data2.reset_index()
                text2 = combine_words(data2)
                D = date_converter(text2)
                print(D[0])
                if D[0] == "Not Found":
                    date =-1
                    print("DATE - NOT FOUND")
                    date_name = "picture_after_date.png"
                    cv2.imwrite(date_name,self.image_frame)
        if D[0] != "Not Found":
            print("DATE -  FOUND")
            date = 1
            if flag == 0:
                datafr = data
            elif flag == 1:
                datafr = data1
            elif flag == 2:
                datafr = data2
            datafr.loc[0, 'position'] = len(datafr.loc[0, 'text'])
            for i in range(1, len(datafr)):
                datafr.loc[i, 'position'] = len(datafr.loc[i, 'text']) + datafr.loc[i - 1, 'position']
            for i in range(len(datafr)):
                if D[1] < datafr.loc[i, 'position']:
                    pos1 = i
                    break
            for i in range(len(datafr)):
                if D[2] <= datafr.loc[i, 'position']:
                    pos2 = i
                    break
            x = datafr.loc[pos1, 'left']
            y = datafr.loc[pos1, 'top']
            x1 = datafr.loc[pos2, 'left']
            y1 = datafr.loc[pos2, 'top']
            w = datafr.loc[pos2, 'width']
            h = datafr.loc[pos2, 'height']
            cv2.rectangle(img, (int(x), int(y)), (int(x1 + w), int(y1 + h)), (0, 255, 0), thickness=3)
            date_name = "picture_after_date.png"
            cv2.imwrite(date_name, img)
        return date


    def take_picture(self, *args):
        original_image = "picture.png"
        cv2.imwrite(original_image, self.image_frame)
        # original_image =cv2.imread('qr.jpg')
        qr = self.qr_detect(self.image_frame)
        logo = self.logo_detect(self.image_frame)
        date = self.date_detect(self.image_frame)
        if date == -1 or logo == -1 or qr == -1:
            final_img = PIL.Image.open("picture_after_date.png")
            size = final_img.size
            #print(size)
            fontsize = size[0] // 40
            draw = ImageDraw.Draw(final_img)
            font = ImageFont.truetype("arial.ttf", fontsize)
            if date == -1:
                date_miss = "MISSING DATE"
                top_rightx = (size[0] * 3) // 4
                top_righty = (size[1] * 1) // 4
                finlal_img = draw.text((top_rightx, top_righty), date_miss, fill=(255, 0, 0), font=font)
            if qr == -1:
                qr_miss = "MISSING QR CODE"
                bottom_leftx = (size[0] * 1) // 4
                bottom_lefty = (size[1] * 3) // 4
                finlal_img = draw.text((bottom_leftx, bottom_lefty), qr_miss, fill=(255, 0, 0), font=font)
            if logo == -1:
                logo_miss = "MISSING LOGO"
                top_leftx = (size[0] * 1) // 4
                top_lefty = (size[1] * 1) // 4
                finlal_img = draw.text((top_leftx, top_lefty), logo_miss, fill=(255, 0, 0), font=font)
            final_img.save('output.jpg')

if __name__ == '__main__':
    MainApp().run()
