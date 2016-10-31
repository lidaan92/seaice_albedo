#!/home/apbarret/builds/anaconda/bin/python
#
import numpy as np
import datetime
import pandas as pd

# Reads the sheba spectra data from Don Perovich and returns two Pandas dataframes: one containing
# information about the samples including date, type of target and general target code; and another containing
# the spectra. 
def load_sheba_spectra():
	
	#fili = "C:/Users/apbarret/Documents/data/Sheba2/ShebaSpectral.csv"
	
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