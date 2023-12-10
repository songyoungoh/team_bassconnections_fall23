import os
import shutil
import random

# Function to sample images and copy to a new folder
def sample_images(source_folder, destination_folder, sample_size=0.1, seed=123):
    # Ensure source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist.")
        return

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    # Filter for images if you want specific formats e.g., JPEG, PNG
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Set the random seed for reproducibility
    random.seed(seed)

    # Determine how many files to sample (10%)
    number_to_sample = max(1, int(len(image_files) * sample_size))
    
    # Randomly sample image files
    sampled_files = random.sample(image_files, number_to_sample)

    # Copy sampled images to the new folder
    for file_name in sampled_files:
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(destination_folder, file_name))
        print(f"Copied {file_name} to {destination_folder}")

# Call the function for both sets of folders with a seed for consistency
sample_images("/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/Train/Flooded", "/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/Test/Flooded", seed = 42)
sample_images("/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/Train/Non-Flooded", "/data/scratch/public/floodnet/team_bassconnections/data/Train_Test_Splits/Test/Non-Flooded", seed = 42)
