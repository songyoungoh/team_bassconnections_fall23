from split_image import split_image
from split_image import split
import os
import shutil
import glob
from PIL import Image
import numpy as np
def splitImages(srcFolder, destFolder, rows, columns) :
    for f1 in os.listdir(srcFolder):
        for f2 in os.listdir(srcFolder+f1):
            for image in os.listdir(srcFolder+f1+"/"+f2):
                print(f1 + f2 + image)
                if f2 == "Flooded":
                    # put original tiles in temporary folder
                    split_image(srcFolder+f1+"/"+f2+"/"+image, rows, columns, should_square=False, should_cleanup=False, output_dir="/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Original_Tiles")
                    # put corresponding mask tiles in temporary folder
                    imageNum = image.split(".")[0]
                    split_image("/shared/data/FloodNet_Dataset/Train/Labeled/Flooded/mask/" + imageNum + "_lab.png", rows, columns, should_square=False, should_cleanup=False, output_dir="/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Mask_Tiles")
                    for maskTile in os.listdir("/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Mask_Tiles"):
                        tile_img = Image.open("/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Mask_Tiles/" + maskTile)
                        np_img = np.array(tile_img) # turn mask into array of pixel classifications
                        original_name = maskTile
                        # Split the string and replace parts as needed
                        parts = original_name.split('_') # split name by at '_'
                        new_name = parts[0] + '_' + parts[2].replace('png', 'jpg') # ?
                        
                        tile_source = "/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Original_Tiles/" + new_name
                        if 1 in np_img:
                            print(maskTile + " contains 1")
                            tile_dest = destFolder+f1+"/Flooded/"
                        else:
                            tile_dest = destFolder+f1+"/Non-Flooded/"
                        
                        # shutil.copy(tile_source, tile_dest)
                        print("copied " + new_name)
                    # clear temp folders for next image   
                    
                    path = '/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Mask_Tiles'
                    for i in os.listdir(path):
                        os.remove(fr'{path}/{i}')
                    print("removed temp masks")
                    path = '/data/scratch/public/floodnet/team_bassconnections/data/TempFolder/Original_Tiles'
                    for i in os.listdir(path):
                        os.remove(fr'{path}/{i}')
                    print("removed temp originals")
                    
                break # for testing just using first 4 images
                '''
                # if non flooded image copy ALL tiles as is
                if f2 == "Non-Flooded":
                    split_image(srcFolder+f1+"/"+f2+"/"+image, rows, columns, should_square=False, should_cleanup=False, output_dir=destFolder+f1+"/"+f2)
                '''
        
#split.split_dir("/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/Test/Flooded/", 2, 2, )

splitImages("/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/", "/data/scratch/public/floodnet/team_bassconnections/data/Segmented_Splits/", 2, 2)