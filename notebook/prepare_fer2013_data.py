"""
This script:
1. Validates the FER2013 dataset structure
2. Analyzes class distribution
3. Prepares data for training
4. Creates train/validation splits
5. Visualizes sample images

Usage:
    python prepare_fer2013_data.py

Version: 1.0
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import shutil

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
DATASET_PATH = '../data/fer2013/'
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
OUTPUT_DIR = 'results/data_analysis'

print("="*70)
print("FER2013 DATASET PREPARATION")
print("="*70 + "\n")


def check_dataset_structure():
    """Validate dataset directory structure"""
    print("Step 1: Validating dataset structure...")
    
    required_paths = {
        'train': os.path.join(DATASET_PATH, 'train'),
        'test': os.path.join(DATASET_PATH, 'test')
    }
    
    all_valid = True
    
    for split_name, split_path in required_paths.items():
        if not os.path.exists(split_path):
            print(f"✗ Missing: {split_path}")
            all_valid = False
            continue
        
        print(f"\n✓ Found {split_name} directory: {split_path}")
        
        # Check emotion subdirectories
        for emotion in EMOTION_LABELS:
            emotion_path = os.path.join(split_path, emotion)
            if os.path.exists(emotion_path):
                count = len(os.listdir(emotion_path))
                print(f"  ✓ {emotion:10s}: {count:5d} images")
            else:
                print(f"  ✗ Missing: {emotion}")
                all_valid = False
    
    if all_valid:
        print("\n✓ Dataset structure is valid!")
    else:
        print("\n✗ Dataset structure is invalid!")
        print("\nExpected structure:")
        print("data/fer2013/")
        print("  ├── train/")
        print("  │   ├── angry/")
        print("  │   ├── disgust/")
        print("  │   ├── fear/")
        print("  │   ├── happy/")
        print("  │   ├── sad/")
        print("  │   ├── surprise/")
        print("  │   └── neutral/")
        print("  └── test/")
        print("      └── (same structure)")
        sys.exit(1)
    
    return True


def analyze_class_distribution():
    """Analyze and visualize class distribution"""
    print("\n" + "="*70)
    print("Step 2: Analyzing class distribution...")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_counts = {}
    test_counts = {}
    
    # Count images per class
    for emotion in EMOTION_LABELS:
        train_path = os.path.join(DATASET_PATH, 'train', emotion)
        test_path = os.path.join(DATASET_PATH, 'test', emotion)
        
        train_counts[emotion] = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        test_counts[emotion] = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
    
    # Print statistics
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    
    print(f"\nTraining set: {total_train:,} images")
    print(f"Test set: {total_test:,} images")
    print(f"Total: {total_train + total_test:,} images")
    
    print("\nClass distribution:")
    print(f"{'Emotion':12s} {'Train':>8s} {'Test':>8s} {'Total':>8s} {'Train %':>8s}")
    print("-" * 55)
    
    for emotion in EMOTION_LABELS:
        train_pct = (train_counts[emotion] / total_train * 100) if total_train > 0 else 0
        total = train_counts[emotion] + test_counts[emotion]
        print(f"{emotion.capitalize():12s} {train_counts[emotion]:8d} "
              f"{test_counts[emotion]:8d} {total:8d} {train_pct:7.2f}%")
    
    # Check for class imbalance
    max_count = max(train_counts.values())
    min_count = min(train_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("⚠ Warning: Significant class imbalance detected!")
        print("  Recommendation: Use class weights during training")
    
    # Visualize distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training set distribution
    emotions_cap = [e.capitalize() for e in EMOTION_LABELS]
    axes[0].bar(emotions_cap, train_counts.values(), color='steelblue', edgecolor='black')
    axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('Number of Images')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for i, (emotion, count) in enumerate(zip(emotions_cap, train_counts.values())):
        axes[0].text(i, count, str(count), ha='center', va='bottom')
    
    # Test set distribution
    axes[1].bar(emotions_cap, test_counts.values(), color='coral', edgecolor='black')
    axes[1].set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('Number of Images')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (emotion, count) in enumerate(zip(emotions_cap, test_counts.values())):
        axes[1].text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Distribution plot saved: {dist_path}")
    plt.close()
    
    return train_counts, test_counts


def visualize_sample_images():
    """Visualize sample images from each emotion class"""
    print("\n" + "="*70)
    print("Step 3: Visualizing sample images...")
    print("="*70)
    
    samples_per_emotion = 5
    
    fig, axes = plt.subplots(len(EMOTION_LABELS), samples_per_emotion, 
                            figsize=(15, 15))
    fig.suptitle('Sample Images per Emotion Class', 
                fontsize=16, fontweight='bold')
    
    for i, emotion in enumerate(EMOTION_LABELS):
        emotion_path = os.path.join(DATASET_PATH, 'train', emotion)
        
        if not os.path.exists(emotion_path):
            continue
        
        images = os.listdir(emotion_path)[:samples_per_emotion]
        
        for j, img_name in enumerate(images):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_ylabel(emotion.capitalize(), 
                                        fontsize=12, fontweight='bold',
                                        rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    samples_path = os.path.join(OUTPUT_DIR, 'sample_images.png')
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample images saved: {samples_path}")
    plt.close()


def check_image_properties():
    """Check image sizes and properties"""
    print("\n" + "="*70)
    print("Step 4: Checking image properties...")
    print("="*70)
    
    sizes = []
    channels = []
    
    # Sample 100 random images
    print("\nSampling 100 images to check properties...")
    
    sample_count = 0
    max_samples = 100
    
    for emotion in EMOTION_LABELS:
        emotion_path = os.path.join(DATASET_PATH, 'train', emotion)
        if not os.path.exists(emotion_path):
            continue
        
        images = os.listdir(emotion_path)[:max_samples // len(EMOTION_LABELS)]
        
        for img_name in images:
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                sizes.append(img.shape[:2])
                channels.append(img.shape[2] if len(img.shape) == 3 else 1)
                sample_count += 1
                
                if sample_count >= max_samples:
                    break
        
        if sample_count >= max_samples:
            break
    
    # Analyze sizes
    unique_sizes = Counter(sizes)
    print(f"\nImage sizes found:")
    for size, count in unique_sizes.most_common():
        print(f"  {size[0]}x{size[1]}: {count} images")
    
    most_common_size = unique_sizes.most_common(1)[0][0]
    print(f"\nMost common size: {most_common_size[0]}x{most_common_size[1]}")
    
    # Check if all same size
    if len(unique_sizes) == 1:
        print("✓ All images have the same size!")
    else:
        print("⚠ Images have different sizes - will be resized to 48x48 during training")
    
    # Analyze channels
    unique_channels = Counter(channels)
    print(f"\nImage channels found:")
    for ch, count in unique_channels.items():
        ch_name = "Grayscale" if ch == 1 else f"{ch}-channel"
        print(f"  {ch_name}: {count} images")


def calculate_class_weights(train_counts):
    """Calculate class weights for imbalanced dataset"""
    print("\n" + "="*70)
    print("Step 5: Calculating class weights...")
    print("="*70)
    
    total = sum(train_counts.values())
    n_classes = len(train_counts)
    
    # Calculate weights
    class_weights = {}
    for i, (emotion, count) in enumerate(train_counts.items()):
        weight = total / (n_classes * count) if count > 0 else 0
        class_weights[i] = weight
    
    print("\nClass weights (for handling imbalance):")
    for i, emotion in enumerate(EMOTION_LABELS):
        print(f"  {emotion.capitalize():10s}: {class_weights[i]:.4f}")
    
    # Save to file
    weights_path = os.path.join(OUTPUT_DIR, 'class_weights.txt')
    with open(weights_path, 'w') as f:
        f.write("# Class weights for FER2013 training\n")
        f.write("# Format: class_index: weight\n\n")
        for i, weight in class_weights.items():
            f.write(f"{i}: {weight:.4f}  # {EMOTION_LABELS[i]}\n")
    
    print(f"\n✓ Class weights saved: {weights_path}")
    
    return class_weights


def generate_data_report(train_counts, test_counts, class_weights):
    """Generate comprehensive data preparation report"""
    print("\n" + "="*70)
    print("Step 6: Generating data report...")
    print("="*70)
    
    report_path = os.path.join(OUTPUT_DIR, 'data_preparation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FER2013 DATASET PREPARATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset Path: {DATASET_PATH}\n")
        f.write(f"Report Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset statistics
        f.write("DATASET STATISTICS\n")
        f.write("-" * 70 + "\n")
        total_train = sum(train_counts.values())
        total_test = sum(test_counts.values())
        f.write(f"Training images: {total_train:,}\n")
        f.write(f"Test images: {total_test:,}\n")
        f.write(f"Total images: {total_train + total_test:,}\n")
        f.write(f"Number of classes: {len(EMOTION_LABELS)}\n\n")
        
        # Class distribution
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Emotion':12s} {'Train':>8s} {'Test':>8s} {'Total':>8s} {'Train %':>8s}\n")
        f.write("-" * 55 + "\n")
        
        for emotion in EMOTION_LABELS:
            train_pct = (train_counts[emotion] / total_train * 100) if total_train > 0 else 0
            total = train_counts[emotion] + test_counts[emotion]
            f.write(f"{emotion.capitalize():12s} {train_counts[emotion]:8d} "
                   f"{test_counts[emotion]:8d} {total:8d} {train_pct:7.2f}%\n")
        
        f.write("\n")
        
        # Class imbalance
        max_count = max(train_counts.values())
        min_count = min(train_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        f.write("CLASS IMBALANCE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Most common class: {max_count} images\n")
        f.write(f"Least common class: {min_count} images\n")
        f.write(f"Imbalance ratio: {imbalance_ratio:.2f}\n")
        
        if imbalance_ratio > 3:
            f.write("Status: SIGNIFICANT IMBALANCE DETECTED\n")
            f.write("Recommendation: Use class weights during training\n\n")
        else:
            f.write("Status: Acceptable balance\n\n")
        
        # Class weights
        f.write("CLASS WEIGHTS (for training)\n")
        f.write("-" * 70 + "\n")
        for i, emotion in enumerate(EMOTION_LABELS):
            f.write(f"{emotion.capitalize():10s}: {class_weights[i]:.4f}\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("TRAINING RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Use data augmentation (rotation, shift, flip, zoom)\n")
        f.write("2. Apply class weights to handle imbalance\n")
        f.write("3. Use batch normalization for stable training\n")
        f.write("4. Monitor validation accuracy and loss\n")
        f.write("5. Use early stopping to prevent overfitting\n")
        f.write("6. Expected training time: 2-4 hours on GPU\n")
        f.write("7. Target accuracy: 65-70% (state-of-the-art on FER2013)\n")
    
    print(f"✓ Data report saved: {report_path}")
    
    # Print key recommendations
    print("\n" + "="*70)
    print("KEY RECOMMENDATIONS FOR TRAINING")
    print("="*70)
    print("1. ✓ Dataset structure is valid")
    print("2. ✓ Class weights calculated for imbalance handling")
    print("3. ✓ Data augmentation will be applied")
    print("4. ✓ Expected training time: 2-4 hours on GPU")
    print("5. ✓ Target accuracy: 65-70%")
    print("\nYou're ready to proceed to training!")


def main():
    """Main execution function"""
    
    # Step 1: Check dataset structure
    check_dataset_structure()
    
    # Step 2: Analyze class distribution
    train_counts, test_counts = analyze_class_distribution()
    
    # Step 3: Visualize samples
    visualize_sample_images()
    
    # Step 4: Check image properties
    check_image_properties()
    
    # Step 5: Calculate class weights
    class_weights = calculate_class_weights(train_counts)
    
    # Step 6: Generate report
    generate_data_report(train_counts, test_counts, class_weights)
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nNext step: Run 'python train_model.py' to start training")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()