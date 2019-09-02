import numpy as np
import sys
import os
import subprocess
import copy
import astropy.io.fits as pyfits
import time

# dr16
master_path = "data.sdss.org/sas/apogeework/apogee/spectro/redux/r12/stars/"

# load catalog
catalog = pyfits.getdata("../allStar-r12-l33.fits")
catalog_id = catalog['APOGEE_ID'].astype("str")
print(catalog_id.shape)

# initiate a batch
batch = int(sys.argv[1])

# loop over all spectra
for i in range(int(2e4)):

    # download spectra
    apogee_id = catalog_id[i+int(batch*2e4)]
    field = catalog['FIELD'][i+int(batch*2e4)]
    loc_id = catalog['LOCATION_ID'][i+int(batch*2e4)]
    filename = 'apStar-r12-%s.fits' % apogee_id.strip() # dr12

    if loc_id == 1:
        filepath = os.path.join(master_path,'apo1m', field.strip(), filename)
    else:
        filepath = os.path.join(master_path,'apo25m', field.strip(), filename)

#    # download spectrum
    os.system("wget --user=sdss --password=2.5-meters " + filepath)
