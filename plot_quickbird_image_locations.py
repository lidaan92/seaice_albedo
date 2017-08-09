def get_image_location(fili, proj4string):

    """
    USAGE: x, y = get_image_center(fili, outproj)

    fili = geotif file
    outproj = proj4 string for output image

    returns: x and y coordinates of center of image
    """

    from osgeo import gdal, osr
    import numpy as np

    ds = gdal.Open(fili)

    prj = ds.GetProjection() # projection of image

    # Get location of image
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    # Calculate center of image
    cntx = gt[0]+(gt[1]*width*0.5)
    cnty = gt[3]+(gt[5]*height*0.5)
    center = np.array([cntx,cnty])
    
    # Calculate extent
    ulxo = gt[0]
    ulyo = gt[3]
    lrxo = gt[0]+gt[1]*width
    lryo = gt[3]+gt[5]*height
    urxo = gt[0]+gt[1]*width
    uryo = gt[3]
    llxo = gt[0]
    llyo = gt[3]+gt[5]*height

    # Convert from image projection to map projection
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(prj)

    outproj = osr.SpatialReference()
    outproj.ImportFromProj4(proj4string)

    ct = osr.CoordinateTransformation(inproj, outproj)
    xstr, ystr, dum = ct.TransformPoint(cntx,cnty)
    ulxm, ulym, dum = ct.TransformPoint(ulxo,ulyo)
    lrxm, lrym, dum = ct.TransformPoint(lrxo,lryo)
    urxm, urym, dum = ct.TransformPoint(urxo,uryo)
    llxm, llym, dum = ct.TransformPoint(llxo,llyo)

#    return xstr, ystr, ulxm, ulym, lrxm, lrym
    return xstr, ystr, (ulxm, urxm, lrxm, llxm), (ulym, urym, lrym, llym)

def draw_image_boundary(bboxx, bboxy, m, color='k', lw=2):

#    from matplotlib.patches import Polygon    

    x = [bboxx[0], bboxx[1], bboxx[2], bboxx[3], bboxx[0]]
    y = [bboxy[0], bboxy[1], bboxy[2], bboxy[3], bboxy[0]]
    m.plot(x, y, lw=lw, color=color)

#----------------------------------------------------------------------
# Main Routine
#----------------------------------------------------------------------
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import numpy as np
import glob

# Get list of images
fileList = glob.glob('QB02*P1BS*.tif')
print fileList

# Set up map and figure
fig = plt.figure(figsize=(12, 12))

#m = Basemap(projection='npstere',boundinglat=67,lon_0=270,resolution='l')
m = Basemap(projection='aea', lat_1=55., lon_0=-130., 
                              llcrnrlon=-140., llcrnrlat=65., 
                              urcrnrlon=-120., urcrnrlat=75., 
                              resolution='h')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

parallels = np.arange(65.,80,5.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(-140.,-120.,5.)
m.drawmeridians(meridians,labels=[True,False,False,True])

# Add map locations
xpt, ypt = m(-122.5,73.)
plt.text(xpt, ypt, 'Banks\nIsland', horizontalalignment='center',
         verticalalignment='center', fontsize=25)

xpt, ypt = m(-135.,74)
plt.text(xpt, ypt, 'Beaufort\nSea', horizontalalignment='center',
         verticalalignment='center', fontsize=30)

xpt, ypt = m(-129.5,67.)
plt.text(xpt, ypt, 'Mackenzie River', horizontalalignment='center',
         verticalalignment='center', rotation=-49., fontsize=18)

for fili in fileList:
    xstr, ystr, bboxx, bboxy = get_image_location(fili,m.proj4string)
    #m.plot(xstr, ystr, marker='D',color='m') # Plot center of image
    draw_image_boundary(bboxx, bboxy, m, color='k', lw=2)

#plt.show()

fig.savefig('/home/apbarret/Documents/Work/Posters/AGU2016/quickbird_image_locations.png')

