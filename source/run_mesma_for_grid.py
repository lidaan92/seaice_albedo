#----------------------------------------------------------------------
# Runs Multiple End-Member Spectral Analysis on a synthetic MODIS image
#
# The code was originally written to produce out for my AGU 2017 poster
# and was run as an IPython/Jupyter notebook.  This notebook was exported 
# a python script.
#
# 2017-08-2014 A.P.Barrett <apbarret@nsidc.org>
#----------------------------------------------------------------------

import seaice_albedo_utilities as util

def validate_config():

    config = {}

    return config

def get_modis_band_refl():
    """
    Reads ice surface spectral libraries and generates MODIS band reflectances

    Returns
    -------
    A tuple of pandas dataframes containing band reflectances
    """
    from spectral.io import envi
    from seaice_albedo_utilities import modis_srf, modis_bbalbedo, spectra2modis
    import re

    from constants import ICE_LIBRARY_PATH, POND_LIBRARY_PATH, OWTR_LIBRARY_PATH

    # Ice
    ice = envi.open(ICE_LIBRARY_PATH)

    # Ponds
    pond = envi.open(POND_LIBRARY_PATH)
    
    # Leads
    lead = envi.open(OWTR_LIBRARY_PATH)

    ## Klugey work around to not include ponds with ice in pond class
    isicey = re.compile('frazil|brash', re.IGNORECASE)
    pond_idx = [i for i, tgt in enumerate(pond.names) if not isicey.search(tgt)]
    # If you want to reverse this you will need to delete the .loc[] for pond_df

    wv, srf = modis_srf()
    ice_df = spectra2modis(ice, wv, srf, code='I')
    pond_df = spectra2modis(pond, wv, srf, code='P')
    lead_df = spectra2modis(lead, wv, srf, code='L')

    return (ice_df, pond_df.iloc[pond_idx], lead_df)

def make_synthetic_grid(ice, pond, lead, igrid, pgrid, ogrid):
    """ Generates grids of synthetic reflectances for a single band"""
    import numpy as np

    return (ice*igrid + pond*pgrid + lead*ogrid + 
            np.random.normal(0.,0.001,igrid.shape))

def get_synthetic_grids(ice, pond, lead, 
                        ice_grid, pond_grid, owtr_grid):
    """
    Returns sythetic band reflectances for bands 1 through 4
    """

    # Select spectra id
    iice, ipnd, iowt = (0, 5, 8)

    band1 = make_synthetic_grid(ice['band1'][iice], pond['band1'][ipnd], 
                                lead['band1'][iowt], 
                                ice_grid, pond_grid, owtr_grid)
    
    band2 = make_synthetic_grid(ice['band2'][iice], pond['band2'][ipnd], 
                                lead['band2'][iowt], 
                                ice_grid, pond_grid, owtr_grid)
    
    band3 = make_synthetic_grid(ice['band3'][iice], pond['band3'][ipnd], 
                                lead['band3'][iowt], 
                                ice_grid, pond_grid, owtr_grid)
    
    band4 = make_synthetic_grid(ice['band4'][iice], pond['band4'][ipnd], 
                                lead['band4'][iowt], 
                                ice_grid, pond_grid, owtr_grid)
    
    return (band1, band2, band3, band4)

def do_unmixing(ice, pond, lead, band1, band2, band3, band4, subset=None):
    """
    Performs spectral unmixing analysis
    """
    import numpy as np
    import numpy.ma as ma
    from seaice_albedo_utilities import find_three_surface

    if subset == None:
        dims = band1.shape
    else:
        dims = subset

    ice_est = np.zeros(dims)
    pond_est = np.zeros(dims)
    owtr_est = np.zeros(dims)
    rmse_est = np.zeros(dims)

    # This is the bread-and-butter part
    for ir in np.arange(0,dims[0]):
        for ic in np.arange(0,dims[1]):

            if ((band1[ic,ir] is not ma.masked) |
                (band2[ic,ir] is not ma.masked) |
                (band3[ic,ir] is not ma.masked) |
                (band4[ic,ir] is not ma.masked)):
                
                print 'Working on %02i %02i' % (ic, ir)
                
                surface = np.array([band1[ic,ir], band2[ic,ir], band3[ic,ir], band4[ic,ir]])
                
                
                farea, rmse = find_three_surface(ice, pond, lead, surface)
                
                best = np.argmin(np.array(rmse))
                ice_est[ic,ir] = farea[best][0]
                pond_est[ic,ir] = farea[best][1]
                owtr_est[ic,ir] = farea[best][2]
                rmse_est[ic,ir] = rmse[best]
            else:
                print 'Cell (%02i %02i) is masked' % (ic, ir)

    return ice_est, pond_est, owtr_est, rmse_est

def make_results_plot(ice_grid, pond_grid, owtr_grid,
                      ice_est, pond_est, owtr_est, figfile=None):

    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(10,5))

    a = f.add_subplot(2,3,1)
    imgplt = plt.imshow(ice_grid, interpolation='none')
    a.set_title('Ice', fontsize=20)
    
    a = f.add_subplot(2,3,2)
    imgplt = plt.imshow(pond_grid, interpolation='none')
    a.set_title('Ponds', fontsize=20)
    
    a = f.add_subplot(2,3,3)
    imgplt = plt.imshow(owtr_grid, interpolation='none')
    a.set_title('Open Water', fontsize=20)
    #plt.colorbar()

    a = f.add_subplot(2,3,4)
    imgplt = plt.imshow(ice_est, interpolation='none')
    a.set_title('Estimated Ice', fontsize=20)
    
    a = f.add_subplot(2,3,5)
    imgplt = plt.imshow(pond_est, interpolation='none')
    a.set_title('Estimated Ponds', fontsize=20)

    a = f.add_subplot(2,3,6)
    imgplt = plt.imshow(owtr_est, interpolation='none')
    a.set_title('Estimated Open Water', fontsize=20)

    plt.tight_layout()

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = f.colorbar(imgplt, cax=cbar_ax)
    cbar.ax.set_ylabel('Fraction', fontsize=20)

    if figfile != None:
        f.savefig(figfile, bbox_extra_artists=(cbar_ax,), bbox_inches='tight')
    else:
        plt.show()

    return 0

def make_rmse_plot(rmse_est, figfile=None):

    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(10,5))
    a = f.add_subplot(1,1,1)
    plt.imshow(rmse_est, interpolation='none')
    a.set_title('RMSE', fontsize=20)
    cbar = plt.colorbar()

    if (figfile != None):
        f.savefig(figfile)
    else:
        plt.show()

    return 0
    

def run_mesma_for_grid():

    from constants import MELTPOND_PATH
    import seaice_albedo_utilities as util
    import numpy as np
    import numpy.ma as ma

    # Get high resolution grid of features
    ice_grid, pond_grid, owtr_grid = util.read_melt_pond_grid(MELTPOND_PATH)

    # Get MODIS band reflectances
    ice, pond, lead = get_modis_band_refl()

    # Generate synthetic grids
    band1, band2, band3, band4 = get_synthetic_grids(ice, pond, lead, 
                                                     ice_grid, pond_grid, owtr_grid)

    # Perform unmixing analysis
    ice_est, pond_est, owtr_est, rmse_est = do_unmixing(ice, pond, lead, 
                                                        band1, band2, band3, band4)
    
    # Mask results
    ice_est = ma.masked_array(ice_est,mask=ice_grid.mask)
    pond_est = ma.masked_array(pond_est,mask=pond_grid.mask)
    owtr_est = ma.masked_array(owtr_est,mask=owtr_grid.mask)
    rmse_est = ma.masked_array(rmse_est,mask=ice_grid.mask)

    make_results_plot(ice_grid, pond_grid, owtr_grid, 
                      ice_est, pond_est, owtr_est, figfile='mesma_results.png')

    make_rmse_plot(rmse_est, figfile='mesma_rmse.png')

    return 0

if __name__ == "__main__":
    run_mesma_for_grid()



