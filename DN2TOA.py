# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:33:06 2019

@author: Rohit Gandikota (NR02440)
"""

import gdal
import sys
import os
#from loguru import logger
import numpy as np
import datetime
import bisect
import math

Earth_Sun_Distance_Table=[

[1,15,32,46,60,74,91,106,121,135,152,166,182,196,213,227,242,258,274,288,305,319,335,349,365],
[.9832,.9836,.9853,.9878,.9909,.9945,.9993,1.0033,1.0076,1.0109,1.0140,1.0158,1.0167,1.0165,1.0149,1.0128,1.0092,1.0057,1.0011,.9972,.9925,.9892,.9860,.9843,.9833]

]

L3_ExoAtmospheric_Irradiance=[181.53,155.92,108.96,24.38]
C2E_MX_ExoAtmospheric_Irradiance=[197.91, 182.37, 157.38, 110.22]


#--- FUNCTIONS ----

def ParseMeta(text_file):
    with open(text_file) as f:
        content = f.readlines()
    return content
def GetEarthSunDistance(time,Earth_Sun_Distance_Table):
    date=time[0:11]
    fdate=datetime.datetime.strptime(date,'%d-%b-%Y')
    dayofyear=fdate.timetuple().tm_yday
    print(dayofyear)
    right_index=bisect.bisect_right(Earth_Sun_Distance_Table[0], dayofyear)
    left_index=right_index-1
    x1=Earth_Sun_Distance_Table[0][left_index]
    x2=Earth_Sun_Distance_Table[0][right_index]
    y1=Earth_Sun_Distance_Table[1][left_index]
    y2=Earth_Sun_Distance_Table[1][right_index]
    x=dayofyear
    y=((y2-y1)/(x2-x1))*(x-x1)+y1
    return (y)

def GenerateTOAImages(irasdir,orasdir,No_Bands,StartOffset,L3_ExoAtmospheric_Irradiance):

    raster_list=[]
    meta_dict={}
    #----------------- FIND ALL INPUT RASTERSIN INPUT LOCATION --------------
    print('Loading Input Rasters and Vectors .....')
    for index in range(StartOffset,No_Bands+StartOffset):
        raster_list.append(irasdir+'/BAND'+str(index)+'.tif')
    content=ParseMeta(irasdir+'/BAND_META.txt')
    content = [x.strip().split('=') for x in content]
    for x in content:
        meta_dict[x[0]]=x[1].strip()
    #print(meta_dict)
    #Required_Keys=['SunElevationAtCenter','B2_Lmin','B3_Lmin','B4_Lmin','B5_Lmin','B2_Lmax','B3_Lmax','B4_Lmax','B5_Lmax','BitsPerPixel','SceneCenterTime']
    index=2
    for im_file in raster_list:
        E0=L3_ExoAtmospheric_Irradiance[index-2]
        image_ds=gdal.Open(im_file)
        Width=image_ds.RasterXSize
        Height=image_ds.RasterYSize
        image_ar=image_ds.ReadAsArray(0,0,Width,Height)
        LMinKey='B'+str(index)+"Lmin"
        LMaxKey='B'+str(index)+"Lmax"
        lmin=float(meta_dict[LMinKey])
        lmax=float(meta_dict[LMaxKey])
        qmin=0
        qmax=2**(int(meta_dict['BitsPerPixel']))-1
        slope=(lmax-lmin)/(qmax-qmin)
        ##yin=lmin
        new_ar=slope*(image_ar-qmin)+lmin  #RADIANCE Array
        #Convert to TOA
        es_distance=GetEarthSunDistance(meta_dict['SceneCentreTime'],Earth_Sun_Distance_Table)
        sunzenith=math.radians(90-float(meta_dict['SunElevationAtCenter']))
        new_ar=(math.pi*(es_distance**2)/E0*math.cos(sunzenith))*new_ar
        #print(new_ar)
        driver = image_ds.GetDriver()
        #print driver
        base_name=os.path.basename(im_file)
        outDs = driver.Create(orasdir+"/"+base_name, Width,Height, 1,gdal.GDT_Float32)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(new_ar)

        outBand.SetNoDataValue(-99)

        # georeference the image and set the projection
        outDs.SetGeoTransform(image_ds.GetGeoTransform())
        outDs.SetProjection(image_ds.GetProjection())
        outBand.FlushCache()

        del outBand,outDs,image_ds,new_ar,image_ar
#----------
#%%
#USAGE
#python DN2TOA.py inputrasdir irasformat outputrasdir
irasdir= 'C:\\Users\\user\\Desktop\\Data\\c2e\\1984813291\\1984813291'
#irasformat=sys.argv[2]
#orasdir=sys.argv[2]
orasdir= 'C:\\Users\\user\\Desktop\\Data\\c2e\\TOA\\1984813291'

GenerateTOAImages(irasdir,orasdir,4,1,C2E_MX_ExoAtmospheric_Irradiance)






    #print(image_ar)
