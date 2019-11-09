# Land-Use-Land-Cover-Classification-of-Satellite-Images-using-Deep-Learning
This work discusses how high resolution satellite images are classified into various classes like cloud, vegetation, water and miscellaneous, using feed forward neural network. Open source python libraries like GDAL and keras were used in this work. This work is generic and can be used for satellite images of any resolution, but with MX band sensors. 

Intially, the raw extracted satellite image has lots of distortions like gemetrical, radiometric etc. We process these images to correct the distortion to the most extent. For this end, we use the code DN2TOA.py.

After generating the TOA images, we used QGIS software to manually clip the pixels from each class and store them as training data. This was the most time consuming process. These clipped pixels have 4 band values per sample. These are the input dimensions and are needed to classified into 4 classes. 

In the code Train_LULC.py, one can find the architecture  of the model used for training. We were able to achieve the following accuracies:
1. Training: 99.87 %
2.Validation: 98.78 %

The code for testing a TOA processed image can be found in Test_MX.py

