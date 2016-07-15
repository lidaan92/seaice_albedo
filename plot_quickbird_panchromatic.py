from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from osgeo import gdal, osr

ds = gdal.Open('QB02_20070609211002_1010010005AD1300_07JUN09211002-P1BS-500489049030_01_P001_u08rf3338.tif')
prj = ds.GetProjection()

# Get location of image
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()

# Calculate center of image
cntx = gt[0]+(gt[1]*width*0.5)
cnty = gt[3]+(gt[5]*height*0.5)

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



