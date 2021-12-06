import numpy as np
import cv2

input_img = 'C:\\Users\\micro\\Desktop\\Gepilatas\\input\\02.png'
output_img = 'C:\\Users\\micro\\Desktop\\Gepilatas\\output\\02.png'
    

# lojuk be kb a zold hatarait

low_green = np.array([21, 52, 32])
high_green = np.array([102, 255, 255])


# kepfeldolgozas
img = cv2.imread(input_img)

if not img:
    raise Exception("Hiba: input img nem talalhato.")

#kicsinyitsunk, hogy hatekonyabb legyen a feldolgozas
img = cv2.resize(img, (900, 650), interpolation=cv2.INTER_CUBIC)


# színek sorrendje BGR, es nem RGB !!!


# 1 --- zold reszek kivagasa

# legyen HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# maszk keszitese
mask = cv2.inRange(imgHSV, low_green, high_green)

#invertaljunk
mask = 255-mask
res1 = cv2.bitwise_and(img, img, mask=mask)


#eredmeny

cv2.imshow("kiindulas", img)
cv2.waitKey(0)
cv2.imshow("maszk", mask)
cv2.waitKey(0)
cv2.imshow('eredmeny', res1)
cv2.waitKey(0)



# 2 --- szurkearnyalatos

greyscale = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    
cv2.imshow('eredmeny', greyscale)
cv2.waitKey(0)

if not cv2.imwrite(output_img,greyscale):
    raise Exception("Hiba: output img nem hozhato letre.")


