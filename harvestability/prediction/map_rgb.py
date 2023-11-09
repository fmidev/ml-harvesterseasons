
from osgeo import osr, ogr, gdal
import os
import argparse



parser = argparse.ArgumentParser(description='Add rbg color to tiff',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--country_name',type=str,help='country name should be provided',required=True)
args = parser.parse_args()

tif_file = f"/home/ubuntu/data/ml-harvestability/predictions/{args.country_name}/{args.country_name}-2023-trfy-r30m-m.tif"



ds = gdal.Open(tif_file, 1)
band = ds.GetRasterBand(1)

# create color table
colors = gdal.ColorTable()

# set color for each value
colors.SetColorEntry(0, (128, 128, 128))
colors.SetColorEntry(1, (0, 97, 0))
colors.SetColorEntry(2, (97, 153, 0))
colors.SetColorEntry(3, (160, 219, 0))
colors.SetColorEntry(4, (255, 250, 0))
colors.SetColorEntry(5, (255, 132, 0))
colors.SetColorEntry(6, (255, 38, 0))
colors.SetColorEntry(7, ( 128, 255, 255 ))
colors.SetColorEntry(8, ( 128, 255, 255 ))
colors.SetColorEntry(9, ( 128, 255, 255 ))

# set color table and color interpretation
band.SetRasterColorTable(colors)
band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

# close and save file
del band, ds



