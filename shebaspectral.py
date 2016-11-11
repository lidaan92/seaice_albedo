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
