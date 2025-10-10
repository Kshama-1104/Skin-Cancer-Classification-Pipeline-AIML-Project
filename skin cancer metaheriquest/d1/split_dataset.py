import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.3, random_seed=42):
    """
    Split dataset into train and validation sets while preserving class structure.
    
    Args:
        source_dir (str): Path to the source directory containing class folders
        train_ratio (float): Ratio of data for training (default: 0.7)
        val_ratio (float): Ratio of data for validation (default: 0.3)
        random_seed (int): Random seed for reproducible splits
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Convert to Path object
    source_path = Path(source_dir)
    
    # Create output directories
    train_dir = Path("train")
    val_dir = Path("val")
    
    # Create train and val directories
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_dirs)} classes:")
    for class_dir in class_dirs:
        print(f"  - {class_dir.name}")
    
    print(f"\nSplitting with {train_ratio*100:.0f}% train and {val_ratio*100:.0f}% validation...")
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all image files in the class directory
        image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        if not image_files:
            print(f"  No image files found in {class_name}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Shuffle the files
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:]
        
        print(f"  Train: {len(train_files)} images")
        print(f"  Validation: {len(val_files)} images")
        
        # Create class directories in train and val
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Move train files
        for file_path in train_files:
            destination = train_class_dir / file_path.name
            shutil.copy2(file_path, destination)
        
        # Move validation files
        for file_path in val_files:
            destination = val_class_dir / file_path.name
            shutil.copy2(file_path, destination)
        
        print(f"  ✓ Completed {class_name}")
    
    print(f"\n✓ Dataset split completed!")
    print(f"Train directory: {train_dir.absolute()}")
    print(f"Validation directory: {val_dir.absolute()}")
    
    # Print summary
    print(f"\nSummary:")
    for split_dir in [train_dir, val_dir]:
        print(f"\n{split_dir.name.upper()}:")
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                file_count = len([f for f in class_dir.iterdir() if f.is_file()])
                print(f"  {class_dir.name}: {file_count} images")

def main():
    """Main function to run the dataset split."""
    
    # Configuration
    source_directory = "a1"  # Source directory containing class folders
    train_ratio = 0.8        # 70% for training
    val_ratio = 0.2          # 30% for validation
    random_seed = 42         # For reproducible results
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory '{source_directory}' not found!")
        print("Please make sure the 'a1' folder exists in the current directory.")
        return
    
    # Validate ratios
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        print("Error: Train and validation ratios must sum to 1.0")
        return
    
    print("=" * 60)
    print("DATASET SPLIT TOOL")
    print("=" * 60)
    print(f"Source directory: {source_directory}")
    print(f"Train ratio: {train_ratio*100:.0f}%")
    print(f"Validation ratio: {val_ratio*100:.0f}%")
    print(f"Random seed: {random_seed}")
    print("=" * 60)
    
    # Confirm before proceeding
    response = input("\nDo you want to proceed with the split? (y/n): ").lower().strip()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    # Perform the split
    try:
        split_dataset(source_directory, train_ratio, val_ratio, random_seed)
    except Exception as e:
        print(f"Error during split: {e}")
        return

if __name__ == "__main__":
    main()
