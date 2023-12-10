from split_image import split_image
from split_image import split
import os
def splitImages(srcFolder, destFolder, rows, columns) :
    for f1 in os.listdir(srcFolder):
        for f2 in os.listdir(srcFolder+f1):
            for image in os.listdir(srcFolder+f1+"/"+f2):
                split_image(srcFolder+f1+"/"+f2+"/"+image, rows, columns, should_square=False, should_cleanup=False, output_dir=destFolder+f1+"/"+f2)
        
splitImages("/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/", "/data/scratch/public/floodnet/team_bassconnections/data/Segmented_Splits/", 3, 3)