import os
import shutil
from tiles_reclassify import splitImages 
from run import run
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def clean_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            os.remove(os.path.join(root, file))

def plot_f1_scores(tile_sizes, f1_scores, save_path):
    sizes = [f'{size}x{size}' for size in tile_sizes]
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, f1_scores, marker='o', linestyle='-')
    plt.xlabel('Tile Size')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Tile Size')
    plt.grid(True)
    
    plt.savefig(save_path)
    
    plt.show()

    plt.close()

def save_confusion_matrix(cm, tile_size, save_dir):
    class_names = ['Flooded', 'Non-Flooded']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix (Tile Size {tile_size}x{tile_size})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    file_name = f'confusion_matrix_{tile_size}x{tile_size}.png'
    save_path = os.path.join(save_dir, file_name)

    plt.savefig(save_path)
    plt.close()



def main(tile_sizes):
    results = []
    f1_scores = []

    for size in tile_sizes:
        print(f"Processing with tile size: {size}x{size}")

        # Process images with the current tile size
        # splitImages(src_folder, dest_folder, size[0], size[1], threshold)

        # Run the model training and evaluation
        data_dir = f"/data/scratch/public/floodnet/team_bassconnections/data/Segmented_Splits/{size}x{size}/"
        accuracy, f1, cm = run(data_dir)

        # Store results for comparison
        results.append((size, accuracy, f1, cm))
        f1_scores.append(f1)
        
        # Save confusion matrix
        save_confusion_matrix(cm, size, "/data/scratch/public/floodnet/team_bassconnections/reports/figures/Confusion_Matrices3")

    # Save the f1 scores
    df = pd.DataFrame(f1_scores)
    df.to_csv("/data/scratch/public/floodnet/team_bassconnections/results/f1_scores.csv", index=False)
    plot_f1_scores(tile_sizes, f1_scores, "/data/scratch/public/floodnet/team_bassconnections/reports/figures/f1_plot3.png")
    
    

if __name__ == "__main__":
    tile_sizes = [1, 2, 3, 4, 6, 8, 12]
    main(tile_sizes)
