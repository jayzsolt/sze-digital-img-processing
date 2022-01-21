import numpy as np
import matplotlib.pyplot as plt
import cv2

# win esetén char escape miatt mindenhol dupla backslash hasznalando.
input_img = 'C:\\Users\\micro\\Documents\\sze-digital-img-processing\\input\\07.png'
output_img = 'C:\\Users\\micro\\Documents\\sze-digital-img-processing\\output\\07.png'

imgTargetWidth = 606
imgTargetHeight = 630

imgTargetWidth = 601
imgTargetHeight = 434

#minIntenzitas = 150
#minIntenzitas = 110
minIntenzitas = 60

binThreshTargetIntenzitas = 255


def refineContours(inputCtrs, minKerulet, maxKerulet):
    retContours = []
    for i in inputCtrs:
        currentKerulet = cv2.arcLength(i,True)
        
        if (currentKerulet >= minKerulet and currentKerulet < maxKerulet):
        # print(cv2.contourArea(i))
        #if (cv2.contourArea(i) >= minTerulet and cv2.contourArea(i) < maxTerulet):
            retContours.append(i)
            
            
    return retContours


def refineContours2(inputCtrs, minMeret, maxMeret, maxTerulet):
    retContours = []
    ctrsBelow5 = 0
    ctrsBelow10 = 0
    ctrsBelow20 = 0
    ctrsBelow50 = 0
    ctrsBelow100 = 0
    ctrsBelow200 = 0
    ctrsBelow500 = 0
    ctrsBelow1000 = 0
    ctrsBelow10000= 0
    for i in inputCtrs:
        x,y,w,h = cv2.boundingRect(i)
#        print("Kontur X width: {}, Y height: {}, korulhatarolo teglalap terulete: {}".format(w, h, (w*h)))

        if w>=minMeret and w<maxMeret and h>=minMeret and h<maxMeret and w*h < maxTerulet:
             retContours.append(i)
                
        if w*h < 5:
             ctrsBelow5 += 1;
                
        if w*h < 10:
             ctrsBelow10 += 1;
                
        if w*h < 20:
             ctrsBelow20 += 1;
                
        if w*h < 50:
             ctrsBelow50 += 1;
                
        if w*h < 100:
             ctrsBelow100 += 1;
                
        if w*h < 200:
             ctrsBelow200 += 1;
                
        if w*h < 500:
             ctrsBelow500 += 1;
                
        if w*h < 1000:
             ctrsBelow1000 += 1;
                
        if w*h < 10000:
             ctrsBelow10000 += 1;
                
    
    print("\n\nKonturok szama: {}\n".format(len(inputCtrs)))
                
    print("Korulhatarolo teglalapok  5  px2 alatt: {} db".format(ctrsBelow5))
    print("Korulhatarolo teglalapok 10  px2 alatt: {} db".format(ctrsBelow10))
    print("Korulhatarolo teglalapok 20  px2 alatt: {} db".format(ctrsBelow20))
    print("Korulhatarolo teglalapok 50  px2 alatt: {} db".format(ctrsBelow50))
    print("Korulhatarolo teglalapok 100 px2 alatt: {} db".format(ctrsBelow100))
    print("Korulhatarolo teglalapok 200 px2 alatt: {} db".format(ctrsBelow200))
    print("Korulhatarolo teglalapok 500 px2 alatt: {} db".format(ctrsBelow500))
    print("Korulhatarolo teglalapok 1000px2 alatt: {} db".format(ctrsBelow1000))
    print("Korulhatarolo teglalapok10000px2 alatt: {} db".format(ctrsBelow10000))
    
    
            
    return retContours        


# lojuk be kb a zold hatarait

low_green = np.array([21, 52, 32])
high_green = np.array([102, 255, 255])


# elo-feldolgozas
img = cv2.imread(input_img)

#kicsinyitsunk, hogy hatekonyabb legyen a feldolgozas
img = cv2.resize(img, (imgTargetWidth, imgTargetHeight), interpolation=cv2.INTER_CUBIC)

cv2.imshow("kiindulas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# színek sorrendje BGR, es nem RGB !!!


# 1 --- zold reszek kivagasa

# legyen HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# maszk keszitese
mask = cv2.inRange(imgHSV, low_green, high_green)

#invertaljunk
mask = 255-mask
res1 = cv2.bitwise_and(img, img, mask=mask)


#eredmenyek

# 2 --- szurkearnyalatos

res2 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)



# 3 --- keressunk alakzatokat


res3=cv2.GaussianBlur(res2,(5,5),1) 
res4=cv2.Canny(res3,10,50) 




partialToShow1 = np.concatenate((res3, res4), axis=1)

cv2.imshow("Zold teruletek nelkul, gauss blur & Konturok", partialToShow1)
cv2.waitKey(0)

res5 = res2
res6 = res2



# kettos kuszoboles, avagy binary threshold, valasszuk ki, milyen intenzitas felett/alatti ertekekkel szeretnenk dolgozni

ret, thresh = cv2.threshold(res3, minIntenzitas, binThreshTargetIntenzitas, cv2.THRESH_BINARY)

cv2.imshow('Kettos kuszoboles (binary threshold) alkalmazva', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()



# hiszterezis kuszoboles?



# nezzuk meg, milyen a kep, ha szincsatornak szerint pl. pirossal dolgozunk

# B, G, R channel splitting
blue, green, red = cv2.split(img)

# detect contours using red channel and without thresholding
contours3, hierarchy3 = cv2.findContours(image=red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# draw contours on the original image
image_contour_red = img.copy()
cv2.drawContours(image=image_contour_red, contours=contours3, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# see the results
cv2.imshow('Konturok - RGB csatornak kozul csak pirosat hasznalva inputkent', image_contour_red)
cv2.waitKey(0)
cv2.destroyAllWindows()





konturok1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# RETR_EXTERNAL NEM JO - az összes belső, nem érintett területet is kiemeli.

img_alap_konturok = img.copy()
cv2.drawContours(img_alap_konturok, konturok1, -1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)





konturok3 = refineContours2(konturok1, 6, 890, 1000000) # itt szurjuk ki a velhetoen nem epulet elemeket

img_konturok_csokkentve2 = img.copy()
cv2.drawContours(img_konturok_csokkentve2, konturok3, -1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

partialToShow5 = np.concatenate((img_alap_konturok, img_konturok_csokkentve2), axis=1)
cv2.imshow('Konturok masodik szures', partialToShow5)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(output_img, img_konturok_csokkentve2)




#
#epsilon = 0.05*cv2.arcLength(i,True) # csak zart alakzatokat keresunk
#            approx = cv2.approxPolyDP(i,epsilon,True)
#
#            if len(approx) == 4:




#
#
#
#
#
#binary = cv2.bitwise_not(res2)
#
#(_,contours,_) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#for contour in contours:
#    (x,y,w,h) = cv2.boundingRect(contour)
#    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
#    
#
#cv2.imshow('eredmeny', img)
#cv2.waitKey(0)



# # approach 3 
#thresh = cv2.threshold(res2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
## Find contours and extract the bounding rectangle coordintes
## then find moments to obtain the centroid
#cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#for c in cnts:
#    # Obtain bounding box coordinates and draw rectangle
#    x,y,w,h = cv2.boundingRect(c)
#    cv2.rectangle(res2, (x, y), (x + w, y + h), (36,255,12), 2)
#
#    # Find center coordinate and draw center point
#    M = cv2.moments(c)
#    cx = int(M['m10']/M['m00'])
#    cy = int(M['m01']/M['m00'])
#    cv2.circle(res2, (cx, cy), 2, (36,255,12), -1)
#    print('Center: ({}, {})'.format(cx,cy))
#
#cv2.imshow('image', res2)
#cv2.waitKey()

# nem jo: ZeroDivisionError: float division by zero




# # approach 3
#
#blur_hor = cv2.filter2D(res2[:, :, 0], cv2.CV_32F, kernel=np.ones((11,1,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
#blur_vert = cv2.filter2D(res2[:, :, 0], cv2.CV_32F, kernel=np.ones((1,11,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
#mask2 = ((res2[:,:,0]>blur_hor*1.2) | (res2[:,:,0]>blur_vert*1.2)).astype(np.uint8)*255
#
#cv2.imshow('eredmeny2', mask2)
#cv2.waitKey(0)

# nem jó: IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed









# approach 4
#
#
#
#def showOpenCVImagesGrid(images, x, y, titles=None, axis="on"):
#    fig = plt.figure()
#    i = 1
#    for image in images:
#        copy = image.copy()
#        channel = len(copy.shape)
#        cmap = None
#        if channel == 2:
#            cmap = "gray"
#        elif channel == 3:
#            copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
#        elif channel == 4:
#            copy = cv2.cvtColor(copy, cv2.COLOR_BGRA2RGBA)
#
#        fig.add_subplot(x, y, i)
#        if titles is not None:
#            plt.title(titles[i-1])
#        plt.axis(axis)
#        plt.imshow(copy, cmap=cmap)
#        i += 1
#    plt.show()
#
#
#def drawLines(image, lines, thickness=1):
#    for line in lines:
#        # print("line="+str(line))
#        cv2.line(image, (line[0], line[1]), (line[2], line[3]),
#                (0, 0, 255), thickness)
#
#
#def drawContours(image, contours, thickness=1):
#    i = 0
#    for contour in contours:
#        cv2.drawContours(image, [contours[i]], i, (0, 255, 0), thickness)
#        area = cv2.contourArea(contour)
#        i += 1
#        
#        
#        
#        
#image = res2
#
#edges = cv2.Canny(image, 50, 200)
#lines = cv2.HoughLinesP(edges, 1, cv2.cv.CV_PI/180, 50, minLineLength=50, maxLineGap=10)[0]
#linesImage = image.copy()
#drawLines(linesImage, lines, thickness=10)
#
#contoursImage = image.copy()
#(contours, hierarchy) = cv2.findContours(res2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#drawContours(contoursImage, contours, thickness=10)
#
#showOpenCVImagesGrid([image, edges, linesImage, contoursImage], 2, 2, titles=["original image", "canny image", "lines image", "contours image"])



### approach 5 - nem lesz jo eredmeny, furan konturozza

##th3 = cv2.adaptiveThreshold(res3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#
#cv2.imshow('Kettos kuszoboles (binary threshold) alkalmazva - adaptiv', th3)
#cv2.waitKey(0)
