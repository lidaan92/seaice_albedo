from osgeo import osr, gdal
import numpy.ma as ma
import numpy as np
import matplotlib
from pylab import *
import glob

fileList = glob.glob('QB02*M1BS*.tif')

for fili in fileList:

    filo = fili.replace('.tif','.png')

    print ('% Printing '+fili+' to '+filo)

    ds = gdal.Open(fili)

    band1 = ds.GetRasterBand(1)
    band2 = ds.GetRasterBand(2)
    band3 = ds.GetRasterBand(3)

    red = band1.ReadAsArray()
    grn = band2.ReadAsArray()
    blu = band3.ReadAsArray()

    image = np.dstack((red, grn, blu))

    fig = figure(figsize=(9,9))
    ax = gca()
    im2 = imshow(image)
    subplots_adjust(bottom=0.09, left=0.11, top=0.94, right=0.99, hspace=0.22)
    savefig(filo, dpi=300)
    close(fig)



