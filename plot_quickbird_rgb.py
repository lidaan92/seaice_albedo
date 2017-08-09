from osgeo import osr, gdal
import numpy.ma as ma
import matplotlib
from pylab import *
#import glob

#fileList = glob.glob()

fili = 'QB02_20070609211002_1010010005AD1300_07JUN09211002-M1BS-500489049030_01_P001_u08rf3338.tif'

ds = gdal.Open(fili)

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

red = band1.ReadAsArray()
grn = band2.ReadAsArray()
blu = band3.ReadAsArray()

image = (0.299*red + 0.587*grn + 0.114*blu)
image = ma.masked_where(image < 1., image)

fig = figure(figsize=(9,9))
ax = gca()
im2 = imshow(image, vmin=0, vmax=255, cmap=cm.gist_gray)
subplots_adjust(bottom=0.09, left=0.11, top=0.94, right=0.99, hspace=0.22)
savefig('QB02_20070609211002_1010010005AD1300_07JUN09211002-M1BS-500489049030_01_P001_u08rf3338.png', dpi=300)
close(fig)



