def read_shebaspectral(fili):
    """
        USAGE: spectra, metadata = read_shebaspectral(filename)

        Reads a text file (shebaspectral_*_misc.txt) of spectral albedos collected
        during the SHEBA campaign.  The date spectra were collected , the number of bands,
        number of targets and a list of target names are returned as a dictionary metadata.  
        Spectra are returned as a numpy array.
    """
    
    f = open(fili,"r")

    # Get header data and strip newline
    h1 = f.readline()
    h2 = f.readline()

    # Get spectra
    spectra = np.genfromtxt(f)

    f.close()

    # Get number of spectra (columns)
    nrow = spectra.shape[0] # Number of rows in spectra: should be 252
    if (len(spectra.shape) > 1):
        ncol = spectra.shape[1]
    else:
        ncol = 1

    # Parse filename to get file date
    dstr = fili.split("/")[-1].split("_")[1]
    year = 1998 # Year is always 1998
    filedate = datetime.datetime.strptime( dstr.strip()+' '+str(year), '%B%d %Y')

    # Parse header strings and create a list of targets
    p = re.compile("j+") # Compile a RE for matching dummy lines in h2
    if not p.match(h2):
        target = [w1.rstrip()+" "+w2.rstrip() for w1, w2 in zip(h1.split("\t"),h2.split("\t"))]
    else:
        target = h1.split("\t") # Might have to use \t+
    
    # Remove empty strings from target list
    target = [w.strip() for w in target if w.strip()]
    
    # Check that number of targets matches number of columns
    if (len(target) != ncol):
        print "****!WARNING: Number of targets in header lines do not match number of columns/spectra****"
    
    # Print results and original strings for checking
    md = {}
    md['source_file'] = fili
    md['filedate'] = filedate #filedate.strftime('%Y-%m-%d')
    md['nband'] = nrow
    md['ntarget'] = ncol
    md['target'] = target
    
    return spectra, md

def read_shebaspectral_csv(fili):

	"""
	USAGE:    date, target, wavelength, spectra = read_shebaspectral_csv(fili)
	
	Reads a csv file containing spectra.
	"""
	
	import datetime
	import numpy as np
	
	f = open( fili, 'r' )
	
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
	tmp = np.genfromtxt(f, delimiter=",")
	
	f.close()
	
	# Put wavelength and spectra into arrays
	wavelength = tmp[:,0]
	spectra = tmp[:,1:]
	
	return date, target, wavelength, spectra

def generate_library_header(spectra, wavelength, target=None, date=None, location=None, name=None):
    
    # Get dimensions of spectra array [0]: wavebands, [1]: spectra
    dims = spectra.shape
    
    dict = {}
    dict["description"] = "NSIDC Sea Ice Spectral Library\n Built by Andrew P. Barrett <apbarret@nsidc.org>\n [2016-11-21]"
    dict["samples"] = dims[1]
    dict["lines"] = dims[0]
    dict["bands"] = 1
    dict["header offset"] = 0
    dict["file type"] = "ENVI Spectral Library"
    dict["data type"] = 4
    dict["interleave"] = "bsq"
    dict["sensor type"] = "Spectron Engineering SE-590"          
    dict["byte order"] = 0
    dict["z plot titles"] = "Wavelength [nm], spectral albedo"
    dict["wavelength units"] = "Nanometers"
    dict["band names"] = "NSIDC Sea Ice Spectral Library"
    dict["reflectance scale factor"] = 100.
    dict["original names"] = target
    dict["spectra dates"] = date
    dict["spectra names"] = name
    dict["spectra locations"] = location
    dict["wavelength"] = wavelength
    
    return dict

def make_spectral_library(spectra, wavelength, target, date, name, filo, idx):
    
    tgt = [target[i] for i in idx]
    dt = [date[i] for i in idx]
    nm = [name[i] for i in idx]
    
    spctr = spectra[:,idx].T
    
    from spectral.io.envi import gen_params
    header = generate_library_header(spctr, wavelength, tgt, dt, nm)
    params = gen_params(header)

    # Write data to spectral library class
    from spectral.io.envi import SpectralLibrary
    specLib = SpectralLibrary(spctr, header, params)

    specLib.save(filo)
	
	
def add_modis_bands(ax=None, facecolor="lightgrey", edgecolor="lightgrey"):

	import pandas as pd
	from matplotlib.patches import Rectangle

	d = {'lower': [620.,841.,459.,545.,1230.,1628.,2105.],'upper': [670.,876.,479.,565.,1250.,1652.,2155.]}
	band = pd.DataFrame(d)
	band['width'] = band['upper']-band['lower']
    
	for bnd in band.index:
		ax.add_patch( Rectangle((band['lower'][bnd], 0.), band['width'][bnd], 1., facecolor=facecolor, edgecolor=edgecolor))
		
