import pooch

url = "ftp://ftp.spacecenter.dk/pub/DTU10/1_MIN/DTU10MDT_1min.nc"
fname = pooch.retrieve(url,"md5:70e2afd3fd7e336ae478b1e740a5f08e")
