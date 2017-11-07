#!/home/apbarret/builds/anaconda/bin/python
#
import numpy as np
import datetime
import pandas as pd

def load_sheba_spectra():
	"""
        Reads the sheba spectra data from Don Perovich and returns two Pandas dataframes: one containing
        information about the samples including date, type of target and general target code; and another containing
        the spectra.
        """

	f = open( "C:/Users/apbarret/Documents/data/Sheba2/ShebaSpectral.csv", 'r' )

	for il in np.arange(0,7):
		f.readline()
    
	# Header lines are at lines 7, 8 and 9
	# Get column headings
	dtstr = f.readline().split(',')
	target1 = f.readline().split(',')
	target2 = f.readline().split(',')

	# Strip first field
	dtstr = dtstr[1:]
	target1 = target1[1:]
	target2 = target2[1:]

	# Get the date and target descriptions (merged into one string) from the header lines
	year=1997
	date = [ datetime.datetime.strptime( d.strip()+' '+str(year), '%d-%b %Y') for d in dtstr ]
	target = [ " ".join((t1.strip(),t2.strip())).strip() for t1,t2 in zip(target1, target2) ]

	nspectra = len(date)

	# Get the spectral albedos from the rest of the file
	tmp= np.genfromtxt(f, delimiter=",")

	f.close()

	# Generate samples dataframe
	samples = pd.DataFrame( {'date': date, 'target': target})
	# Add a target code to samples
	samples['code'] = 'M'
	samples['code'].loc[samples['target'].str.contains('pond|mp')] = 'P'
	samples['code'].loc[samples['target'].str.contains('White ice')] = 'I'
	samples['code'].loc[samples['target'].str.contains('lead')] = 'L'


	# Generate spectra dataframe
	spectra = pd.DataFrame( tmp[:,1:], index=tmp[:,0] )
	
	return (samples, spectra)

# Loads MODIS spectral response functions
def modis_srf():

	"""
	USAGE: wavelength, srf = modis_srf()

	Loads MODIS spectral response functions from MODIS.f

        Returns: a tuple containing a list of wavelengths for the filter function and
                 the weights of the filter function
	"""
	from constants import MODIS_SRF_PATH
 	
	import re
	p0 = re.compile( '      DATA' )
	p1 = re.compile( '     A.*,')
	p2 = re.compile( '     A.*/' )

	nw = 1501

	band = 0

	rsr = {}

	f = open( MODIS_SRF_PATH, "r" )
	lines = f.readlines()
	for l in lines:
		if p0.match( l ):
			wgt = np.zeros(nw) # Initialize weight function
			m = re.search('(\d+)\*0\.,', l)
			i0 = int(m.group(1)) # Extract index for start of non-zero values in function
		if p1.match( l ):
			val = l[6:].strip().split(',') # Extract values
			ww = [ float(v) for v in val[0:len(val)-1] ] # Convert to float
			i1 = i0+len(ww)
			wgt[i0:i1] = ww # Insert into array
			i0 = i1
		if p2.match( l ):
			rsr[band] = wgt # Add array to dictionary
			band += 1
	f.close()

	# Define wavelengths for filter function
	filtwv = np.arange(250,4000.1,2.5)
	
	return filtwv, rsr
	
def modis_bbalbedo( spwv, spal, flwv, filt ):

	"""
	USAGE: albedo = modis_bbalbedo( spectra_wavelength, spectra_reflectance, srf_wavelength, srf )
	
	where srf is spectral response function
	
	Calculates MODIS band albedos
	"""
        
    # Subset spec. res. func. wavelength and responses to nonzero elements
	idx = np.nonzero(filt)
	xflwv = flwv[idx]
	yfilt = filt[idx]
    
    # Interpolate spectrometer spectral albedo to modis SRF wavelengths
	spalint = np.interp(xflwv, spwv, spal)
    
    # Calculate broadband albedo as weighted sum of spectral albedo and
    # filter response function
	modr = sum(yfilt*spalint)/sum(yfilt)

	return(modr)

def spectra2modis(specLib, wv, srf, code=None):
    """
    Converts a spectral library to a dataframe containing MODIS band albedos

    USAGE: df = sepctra2modis(specLib, wv, srf, code=None)

    specLib - a spectral library object returned by spectral.io.envi.open()
    wv      - list of wavelengths returned by modis_srf()
    srf     - list of spectral response function weights returned by modis_srf()
    code    - optional code to assign to band albedos

    Returns: pandas dataframe containing band albedos
    """
    
    from seaice_albedo_utilities import modis_srf, modis_bbalbedo
    import re
    import datetime as dt
    
    # Parse spectra names
    sftype_ptrn = re.compile('^\w+ ')
    trgt_ptrn = re.compile('(?<=\[)(\w+ *){0,}')
    date_ptrn = re.compile('(\d{8})')
    sftype = [sftype_ptrn.search(a).group(0) for a in specLib.names]
    target = [trgt_ptrn.search(a).group(0) for a in specLib.names]
    date = [dt.datetime.strptime(date_ptrn.search(a).group(0),'%Y%m%d') for a in specLib.names]
    
    # put into a pandas dataframe
    import pandas as pd
    df = pd.DataFrame({'sftype': sftype, 'target': target, 'date': date, 'code':code})
    
    # Calculate band albedos
    df['band1'] = [modis_bbalbedo(specLib.bands.centers, specLib.spectra[i,:], wv, srf[0]) for i in np.arange(0,specLib.spectra.shape[0])] 
    df['band2'] = [modis_bbalbedo(specLib.bands.centers, specLib.spectra[i,:], wv, srf[1]) for i in np.arange(0,specLib.spectra.shape[0])] 
    df['band3'] = [modis_bbalbedo(specLib.bands.centers, specLib.spectra[i,:], wv, srf[2]) for i in np.arange(0,specLib.spectra.shape[0])] 
    df['band4'] = [modis_bbalbedo(specLib.bands.centers, specLib.spectra[i,:], wv, srf[3]) for i in np.arange(0,specLib.spectra.shape[0])]
    
    return df
	
def spectral_mixture_solver(ice, pond, lead, b):
    
    """
    USAGE: a, resid, rmse, isrange, isrmse, isresid = spectral_mixture_solver(ice, pond, lead, target)
    
    where 
    ice, pond, and lead: are n-element vectors with band reflectances
    target: vector containing band reflectances of target
    
    Returns:
        a - area fraction weights
        resid - vector of residuals
        rmse - scalar of root mean squared error
        isrange - boolean showing elements of a are within acceptable range [-0.01, 1.01]
        isrmse - boolean showing rmse is < 2.5% (0.025)
        isresid - boolean showing no three consecutive residuals are > 2.5% (0.025)
        
    Description:
        Algorithm is based on Painter's MODSCAG algorithm
        
    """
    
    # Put spectra into matrix
    A =np.vstack((np.hstack((ice,1)), 
                  np.hstack((pond,1)), 
                  np.hstack((lead,1)))).T
    ba = np.hstack((b,1))
    
    # Solve matrix - uses Gram-Schmidt QR
    m = np.linalg.lstsq(A, ba)[0]
    
    # Calculate estimates
    yhat = np.dot(A,m)
    
    # Calculate residual
    resid = ba - yhat
    
    # calculate rmse
    rmse = np.linalg.norm(resid)*0.5 # RMSE is half the frobius norm
    
    return (m, resid, rmse)

def find_three_surface(ice_df, pond_df, lead_df, surface):
	
    """
    USAGE: area_frac, rmse = find_three_surface(ice_df, pond_df, lead_df, surface)

    ice_df, pond_df, lead_df - dataframes containing MODIS band albedos
    
    """
	
    nice = ice_df['code'].count()
    npond = pond_df['code'].count()
    nlead = lead_df['code'].count()
    
    # Find best three surface solution
    area_frac = []
    rmse_list = []
    for iceid in np.arange(0,nice):
        for pndid in np.arange(0,npond):
            for ledid in np.arange(0,nlead):
            
                a, resid, rmse = spectral_mixture_solver(ice_df[['band1',
                                                                 'band2',
                                                                 'band3',
                                                                 'band4']].iloc[iceid].values,
							 pond_df[['band1',
                                                                  'band2',
                                                                  'band3',
                                                                  'band4']].iloc[pndid].values,
                                                         lead_df[['band1',
                                                                  'band2',
                                                                  'band3',
                                                                  'band4']].iloc[ledid].values,
                                                         surface)
    
                isrange = np.all((a > -0.01) & (a < 1.01))
                isresid = np.all((resid[0:3] < 0.025) & (resid[1:4] < 0.025) & (resid[2:] < 0.025))
                isrmse = rmse < 0.025
                isvalid = np.all((isrange,isresid,isrmse))
        
                if (isvalid):
                        area_frac.append(a)
                        rmse_list.append(rmse)
            
    return (area_frac, rmse_list)

def read_melt_pond_grid(fili):
    """
    Reads grid of melt pond statistics from NOAA Beaufort sea melt pond data set 
    
    Fetterer, F., S. Wilds, and J. Sloan. 2008. Arctic sea ice melt pond statistics and maps, 1999-2001, 
    [list the dates of the data used]. Boulder, Colorado USA: National Snow and Ice Data Center. 
    http://dx.doi.org/10.7265/N5PK0D32
    
    Argument
    --------
    fili - filepath
    
    Returns
    -------
    tuple of grids giving fractions of ice, pond and open water
    """
    
    import numpy.ma as ma
    import numpy as np
    import re
    import urllib

    if (re.match("ftp:",fili)):
        f = urllib.urlopen(fili)
    else:
        f = open(fili, 'r')

    # Read through header
    for il in np.arange(0,3):
        f.readline()
    
    # Read column headings. 
    header = f.readline()

    tmp = np.genfromtxt(f)

    f.close()

    row = np.int8(np.mod(tmp[:,0],100.))
    col = np.int8(np.floor(tmp[:,0]/100))

    ice_grid = tmp[:,1].reshape(20,20)
    pond_grid = tmp[:,2].reshape(20,20)
    owtr_grid = tmp[:,3].reshape(20,20)

    # Mask grids
    ice_grid = ma.masked_where(ice_grid > 9., ice_grid)
    pond_grid = ma.masked_where(pond_grid > 9., pond_grid)
    owtr_grid = ma.masked_where(owtr_grid > 9., owtr_grid)

    return (ice_grid, pond_grid, owtr_grid)
