import os

# Path for pond data to use to generate synthetic image
MELTPOND_PATH = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02159/G02159_Cell_coverages_txt_files/beaufo_2001sep03a_3c_stats_v1.0.txt"

# Spectral library file paths
LIBRARY_DIRI = os.path.join('..','data')
#LIBRARY_DIRI = "/disks/arctic5_raid/abarrett/ShebaSpectral"
ICE_LIBRARY_FILE = "Sheba.SpectralLibrary.ice.filled.hdr"
POND_LIBRARY_FILE = "Sheba.SpectralLibrary.pond.filled.hdr"
OWTR_LIBRARY_FILE = "Sheba.SpectralLibrary.lead.filled.hdr"
SNOW_LIBRARY_FILE = "Sheba.SpectralLibrary.snow.filled.hdr"

ICE_LIBRARY_PATH = os.path.join(LIBRARY_DIRI,ICE_LIBRARY_FILE)
POND_LIBRARY_PATH = os.path.join(LIBRARY_DIRI,POND_LIBRARY_FILE)
OWTR_LIBRARY_PATH = os.path.join(LIBRARY_DIRI,OWTR_LIBRARY_FILE)
SNOW_LIBRARY_PATH = os.path.join(LIBRARY_DIRI,SNOW_LIBRARY_FILE)

# MODIS spectral response function file path
#MODIS_SRF_DIRI = "/disks/arctic5_raid/abarrett/ShebaSpectral"
MODIS_SRF_DIRI = os.path.join('..','data')
MODIS_SRF_FILE = "MODIS.f"
MODIS_SRF_PATH = os.path.join(MODIS_SRF_DIRI, MODIS_SRF_FILE)


