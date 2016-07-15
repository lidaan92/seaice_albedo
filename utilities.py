from osgeo import gdal, osr

# Extracts the coordinates of an image in the native projection and
# returns a 2D arrays of these coordinates.
def GetImageLocation(fili, corner=True):

    # Defines pixel offset if image corners are
    # requested
    if (corner = True) then:
        xshift = 0.5
        yshift = 0.5
    else:
        xshift = 0.0
        yshift = 0.0

    # Open file
    ds = gdal.Open(fili)

    # Get image projection
    proj = ds.GetProjection()

    # Define WGS84 Platte Caree
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
           SPHEROID["WGS 84",6378137,298.257223563,
               AUTHORITY["EPSG","7030"]],
           AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
           AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
           AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""

    # Get location and size of image
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    # Calculate corners of image

    # Calculate center of image
    cntx = gt[0]+(gt[1]*width*0.5)
    cnty = gt[3]+(gt[5]*height*0.5)

    # Put coordinates into some structure

    # Generate projection tranformation
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)

    outproj = osr.SpatialReference()
    outproj.ImportFromWkt(wgs84_wkt)

    transform = osr.CoordinateTransformation(inproj,outproj)

    lonlat = transform.TransformPoint(cntx,cnty)

    # Return structure
    return lonlat






