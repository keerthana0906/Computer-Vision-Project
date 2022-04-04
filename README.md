# Computer-Vision-Project
Mobile App for QR Code , Date and logo detection done as a part of Computer Vision course

QR code : Missing QR code is detected using the inbuilt QRCodeDetector() present cv2 library   
Logo    : Used SIFT feature matching. Few logos are priorly stored in a folder. SIFT feature matching is done for the input image with every logo present in the folder. If the none of the logo is passing the minimun match count then the logo will be marked missing  
Date    : Pytessaract is used to extract the text from image and regex are used to see if there is any date present in that text extracted.  

Kivy has been used as part of mobile app development.
