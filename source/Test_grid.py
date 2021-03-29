import numpy as np
from math import *
from tqdm import  tqdm
# #%%Test
# import matplotlib.pyplot as plt

# plt.plot([1, 2, 3, 4])
# plt.ylabel('y-label')
# plt.show()


#%%
# import cv2
# import matplotlib.pyplot as plt
# inFileName  = "./dataset/91/000t1.bmp"

# img=cv2.imread(inFileName)

# imgGray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#%%

im_height_org = 3072
im_width_org = 3072

# im_height_org = 10
# im_width_org = 10


# gw = 10
# gh = 10

gw = im_width_org
gh = im_height_org

imgSize = gw * gh
initValue = 100

t1_w = 2
gw_total = 2.3

t2_w = gw_total - t1_w

t1_rate = 0.02
t2_rate = -0.03
y_start = 0

degree = 0.1
m_startValue = 0.0 

arr_rate = np.full((gh,gw),1,dtype=np.float32)

#%%



for y in range(gh):
    
    startPos = -(y+y_start)*tan(radians(degree))
    
    m_startValue = (int) (startPos / gw_total)
    
    if m_startValue !=0:
        gCnt = - m_startValue
    else:
        gCnt = 0 
         
    for x in range(gw):    
        
        g_sp = startPos + gCnt*gw_total
        
        if g_sp > x:
            g_sp = g_sp - gw_total
            
        if g_sp <=x:
            
            #t1 process 
            pt1_cal_pos = g_sp + t1_w
            
            if pt1_cal_pos <=x+1:
                if g_sp <=x:
                    pxT1rate = pt1_cal_pos -x
                    
                    if pxT1rate < 0 :
                        pxT1rate =0 
                else:
                    pxT1rate = pt1_cal_pos - g_sp
                    pxT2rate = g_sp -x
                    
            else:
                if g_sp <=x:
                    pxT1rate =1
                    pxT2rate =0
                else:
                    pxT1rate = x+1 - g_sp
                    pxT2rate = g_sp -x 
                    
            #t2 process
            
            pt2_cal_pos = pt1_cal_pos+t2_w
            
            if pt2_cal_pos <= x+1:
                if pt1_cal_pos <=x:
                    pxT2rate = pt2_cal_pos -x 
                else:
                    pxT2rate = pt2_cal_pos - pt1_cal_pos
                    
                pxT1rate = pxT1rate +x+1 -pt2_cal_pos
            
            else:
                if pt1_cal_pos <=x :
                    pxT2rate = 1
                    pxT1rate = 0 
                    
                else:
                        if pt1_cal_pos <=x+1:
                            pxT2rate = x+1 -pt1_cal_pos
                            
        else:
            pxT2rate = 0.5 
            pxT1rate = 0.5 
            
        
        if g_sp + gw_total < x+1 and g_sp <=x :
            gCnt +=1
        
        if x==10:
            a=0.5
            
        arr_rate[y][x] = (
            pxT1rate + (pxT1rate*t1_rate) + 
                pxT2rate +(pxT2rate*t2_rate)
            )
        
        # arr_rate[y][x] = arr_rate[y][x] *(
        #     pxT1rate + (pxT1rate*t1_rate) + 
        #         pxT2rate +(pxT2rate*t2_rate)
        #     )
            

                        
#%%   input value



def getRate(value):
    return 0.000000589*value -0.0000619    

# arr_rate = img






def createImageWidthGrid( inImg,w,h,
                          t1Rate,t2Rate,t1w,twTotal
                          ,angle,isH,isGrid=1):
    # t1_rate = 30
    # t2_rate = -5
    
    t1_rate = t1Rate
    t2_rate = t2Rate
    y_start = 0
    
    degree = angle
    
    if(isH == 1): # vertical 90 degree
        gw = h
        gh = w   
    else:
        gw = w
        gh = h   
        
    
    # t1_w = 1.6
    # gw_total = 2.3
    
    t1_w = t1w
    gw_total=twTotal
    
    t2_w = gw_total - t1_w
    
    m_startValue = 0.0 
    
    if inImg.dtype !=np.float32:
        inImg = inImg.astype(np.float32)
    
    # for y in tqdm( range(gh)):
    for y in range(gh):
    
        startPos = -(y+y_start)*tan(radians(degree))
        
        m_startValue = (int) (startPos / gw_total)
        
        if m_startValue !=0:
            gCnt = - m_startValue
        else:
            gCnt = 0 
             
            
        ranT1= np.random.normal(loc=t1_rate, scale=1.0, size=gw)
        ranT2= np.random.normal(loc=t2_rate, scale=1.0, size=gw)
    
        for x in range(gw):    
            
            g_sp = startPos + gCnt*gw_total
            
            if g_sp > x:
                g_sp = g_sp - gw_total
                
            if g_sp <=x:
                
                #t1 process 
                pt1_cal_pos = g_sp + t1_w
                
                if pt1_cal_pos <=x+1:
                    if g_sp <=x:
                        pxT1rate = pt1_cal_pos -x
                        
                        if pxT1rate < 0 :
                            pxT1rate =0 
                    else:
                        pxT1rate = pt1_cal_pos - g_sp
                        pxT2rate = g_sp -x
                        
                else:
                    if g_sp <=x:
                        pxT1rate =1
                        pxT2rate =0
                    else:
                        pxT1rate = x+1 - g_sp
                        pxT2rate = g_sp -x 
                        
                #t2 process
                
                pt2_cal_pos = pt1_cal_pos+t2_w
                
                if pt2_cal_pos <= x+1:
                    if pt1_cal_pos <=x:
                        pxT2rate = pt2_cal_pos -x 
                    else:
                        pxT2rate = pt2_cal_pos - pt1_cal_pos
                        
                    pxT1rate = pxT1rate +x+1 -pt2_cal_pos
                
                else:
                    if pt1_cal_pos <=x :
                        pxT2rate = 1
                        pxT1rate = 0 
                        
                    else:
                            if pt1_cal_pos <=x+1:
                                pxT2rate = x+1 -pt1_cal_pos
                            else:
                                pxT2rate =0
                            
                

                                
                            
                                
            else:
                pxT2rate = 0.5 
                pxT1rate = 0.5 
                
            
            if g_sp + gw_total < x+1 and g_sp <=x :
                gCnt +=1
            
                
            # arr_rate[y][x] = (
            #     pxT1rate + (pxT1rate*t1_rate) + 
            #         pxT2rate +(pxT2rate*t2_rate)
            #     )


            if(isH == 1):
                inVal = inImg[x][y]
                                
                # if isGrid ==1:
                #     prate= getRate(inVal)
                # else:
                prate = 0.02
                
                calOut = inVal*(
                    pxT1rate + (pxT1rate*ranT1[x]*prate) + 
                        pxT2rate +(pxT2rate*ranT2[x]*prate)
                    )
                
                inImg[x][y] = calOut
            else:     
                
                inVal = inImg[y][x]
                prate= getRate(inVal)
                
                # if isGrid ==1:
                #     prate= getRate(inVal)
                # else:
                prate = 0.02
                                
                calOut= inImg[y][x] *(
                    pxT1rate + (pxT1rate*ranT1[x]*prate) + 
                        pxT2rate +(pxT2rate*ranT2[x]*prate)
                    )
                inImg[y][x] = calOut
            
    
    
    return inImg
                


#%%
# import matplotlib.pyplot as plt
# from pydicom import dcmread
# from pydicom.data import get_testdata_file

# # fpath = get_testdata_file('CT_small.dcm')
# # fpath = '1417_103_r8_120cm_Long_Chest_1.dcm'
# fpath = 'Before grdi suppression chest phantom.dcm'
# ds = dcmread(fpath)

# # Normal mode:
# print()
# print(f"File path........: {fpath}")
# print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
# print()

# pat_name = ds.PatientName
# display_name = pat_name.family_name + ", " + pat_name.given_name
# print(f"Patient's Name...: {display_name}")
# print(f"Patient ID.......: {ds.PatientID}")
# print(f"Modality.........: {ds.Modality}")
# print(f"Study Date.......: {ds.StudyDate}")
# print(f"Image size.......: {ds.Rows} x {ds.Columns}")
# print(f"Pixel Spacing....: {ds.PixelSpacing}")

# # use .get() if not sure the item exists, and want a default value if missing
# print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

# # plot the image using matplotlib
# plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
# plt.show()

# imgShape = ds.pixel_array.shape

#%%

# import matplotlib.pyplot as plt

# inFileName = "201029_133749_3072x3072_0001_f.raw"

# img=np.fromfile(inFileName ,dtype=np.uint16).reshape(im_height_org,im_width_org)


#%% code test
# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pydicom import dcmread
# from pydicom.data import get_testdata_file

# from PIL import Image

# # fpath = get_testdata_file('CT_small.dcm')
# fpath = '56,200,25_Vieworks1717_3072x3072.raw'
# # ds = dcmread(fpath)
# # dcmImg= ds.pixel_array
# # imgW= ds.Columns
# # imgH= ds.Rows

# imgW= 3072
# imgH= 3072

# imgNp= np.fromfile(fpath,dtype=np.uint16).reshape(imgW,imgH)

# # imgT = imgNp[i:i + image_size, j:j + image_size]
# gw=100
# gh=100

# i= 789
# j=1284
# imgT = imgNp[i:i + gw, j:j + gh]

# # out_image = np.zeros((gw, gh, 1), dtype=np.uint16)
# # img = out_image+100
# img = imgT

# # retImg = img*arr_rate

# arr_rate = np.copy(img)

# angle_val = 1.5

# isHorizontal = 0

# t1_rate= 2
# t2_rate = -3

# t1_with = 1.3
# tw_total = 2.7

# retImg = createImageWidthGrid(arr_rate ,gw,gh,
#                               t1_rate,t2_rate,t1_with,tw_total,
#                               angle_val,isHorizontal)

# retImgOut = retImg.astype(np.uint16)

# # retImgOut.tofile("test_grid_100x100_01.raw")

# # with open(wfileName ,"wb") as f:    
# #     f.write(img_out)



#%% test random function 
# import matplotlib.pyplot as plt


# rand_t = np.random.normal(loc=2.0, scale=1.0, size=100)

# # rand_t = np.random.standard_t(df=3, size=100)


# count, bins, ignored = plt.hist(rand_t, bins=20)


#%%

# import cv2
# import matplotlib.pyplot as plt
# inFileName  = "./dataset/91/000t1.bmp"

# img=cv2.imread(inFileName)

# imgGray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# img = imgGray

# h,w= img.shape

# gw = w
# gh = h