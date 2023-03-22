#!/bin/sh

if [ -d "maps" ]
then 
    echo "Directory maps exists."
else
    echo "Directory maps is created."
    mkdir maps
fi

wget https://www.dropbox.com/s/i45oz664elds9h5/mask_gal_ns1024_equ.fits -P ./maps
wget https://www.dropbox.com/s/w32jwzc78tudyht/mask_sync_ns1024_equ.fits -P ./maps
