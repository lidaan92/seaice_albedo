from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import numpy as np

ds = gdal.Open('QB02_20070609211002_1010010005AD1300_07JUN09211002-P1BS-500489049030_01_P001_u08rf3338.tif')
prj = ds.GetProjection()

# get data
data = ds.ReadAsArray()
# sub-sample data for display
img = data[::5,::5]

# Get location of image
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
print gt


# Calculate center of image
cntx = gt[0]+(gt[1]*width*0.5)
cnty = gt[3]+(gt[5]*height*0.5)
center = np.array([cntx,cnty])

# Calculate extent
ulxo = gt[0]
ulyo = gt[3]
lrxo = gt[0]+gt[1]*width
lryo = gt[3]+gt[5]*height

# Set up map and figure
fig = plt.figure(figsize=(12, 12))

m = Basemap(projection='npstere',boundinglat=60,lon_0=270,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

# Convert from image projection to map projection
inproj = osr.SpatialReference()
inproj.ImportFromWkt(prj)

outproj = osr.SpatialReference()
outproj.ImportFromProj4(m.proj4string)

ct = osr.CoordinateTransformation(inproj, outproj)
xstr, ystr, dum = ct.TransformPoint(cntx,cnty)
ulxm, ulym, dum = ct.TransformPoint(ulxo,ulyo)
lrxm, lrym, dum = ct.TransformPoint(lrxo,lryo)

m.plot(xstr, ystr, marker='D',color='m')
m.plot(ulxm, ulym, marker='D',color='r')
m.plot(lrxm, lrym, marker='D',color='r')

#plt.imshow(img,'Greys')

plt.show()



