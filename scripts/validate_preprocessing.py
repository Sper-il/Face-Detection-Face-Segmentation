"""
Quick validation script to check preprocessing outputs.
Run after preprocessing to ensure data quality.
"""

import json
from pathlib import Path
import sys


def validate_wider_face():
    """Validate WIDER FACE preprocessing outputs."""
    print("\n📊 Validating WIDER FACE Dataset...")
    print("-" * 60)
    
    base = Path("e:/Face Detection & Face Segmentation/data/processed/wider_face")
    
    issues = []
    
    # Check annotations file
    ann_file = base / "annotations" / "wider_face_annotations.json"
    if not ann_file.exists():
        issues.append("❌ Annotations file not found")
        return issues
    
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    if len(annotations) == 0:
        issues.append("❌ No annotations found")
    else:
        print(f"✅ Found {len(annotations)} annotated images")
        
        # Count faces
        total_faces = sum(len(sample['faces']) for sample in annotations)
        print(f"✅ Total faces: {total_faces}")
        
        # Check splits
        train_count = sum(1 for s in annotations if s['split'] == 'train')
        val_count = sum(1 for s in annotations if s['split'] == 'val')
        print(f"✅ Train: {train_count}, Val: {val_count}")
    
    # Check statistics
    stats_file = base / "statistics" / "wider_face_stats.json"
    if not stats_file.exists():
        issues.append("⚠️ Statistics file not found")
    else:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"✅ Statistics generated")
        print(f"   Retention rate: {stats['summary']['retention_rate_images']}")
    
    # Check images directory
    img_dir = base / "images"
    if img_dir.exists():
        img_count = len(list(img_dir.glob("*.jpg")))
        print(f"✅ Images directory: {img_count} files")
        
        if img_count != len(annotations):
            issues.append(f"⚠️ Image count ({img_count}) != annotation count ({len(annotations)})")
    else:
        issues.append("❌ Images directory not found")
    
    # Check visualizations
    viz_dir = base / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        print(f"✅ Visualizations: {len(viz_files)} files")
    else:
        issues.append("⚠️ Visualizations not generated")
    
    return issues


def validate_celebamask():
    """Validate CelebAMask-HQ preprocessing outputs."""
    print("\n📊 Validating CelebAMask-HQ Dataset...")
    print("-" * 60)
    
    base = Path("e:/Face Detection & Face Segmentation/data/processed/celebamask_hq")
    
    issues = []
    
    # Check splits
    for split in ['train', 'val', 'test']:
        img_dir = base / split / "images"
        mask_dir = base / split / "masks"
        
        if not img_dir.exists():
            issues.append(f"❌ {split}/images not found")
            continue
        
        if not mask_dir.exists():
            issues.append(f"❌ {split}/masks not found")
            continue
        
        img_count = len(list(img_dir.glob("*.jpg")))
        mask_count = len(list(mask_dir.glob("*.png")))
        
        if img_count != mask_count:
            issues.append(f"❌ {split}: image count ({img_count}) != mask count ({mask_count})")
        else:
            print(f"✅ {split.upper()}: {img_count} image-mask pairs")
    
    # Check statistics
    stats_file = base / "statistics" / "celebamask_hq_stats.json"
    if not stats_file.exists():
        issues.append("⚠️ Statistics file not found")
    else:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"✅ Statistics generated")
        print(f"   Total valid: {stats['summary']['valid_images']}")
        print(f"   Retention: {stats['summary']['retention_rate']}")
    
    # Check sample list
    samples_file = base / "celebamask_hq_samples.json"
    if not samples_file.exists():
        issues.append("⚠️ Sample list not found")
    else:
        with open(samples_file, 'r') as f:
            samples = json.load(f)
        print(f"✅ Sample list: {len(samples)} entries")
    
    # Check visualizations
    viz_dir = base / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        print(f"✅ Visualizations: {len(viz_files)} files")
    else:
        issues.append("⚠️ Visualizations not generated")
    
    return issues


def main():
    """Main validation routine."""
    print("=" * 60)
    print("🔍 Data Preprocessing Validation")
    print("=" * 60)
    
    all_issues = []
    
    # Validate WIDER FACE
    wider_issues = validate_wider_face()
    all_issues.extend(wider_issues)
    
    # Validate CelebAMask-HQ
    celeb_issues = validate_celebamask()
    all_issues.extend(celeb_issues)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Validation Summary")
    print("=" * 60)
    
    if not all_issues:
        print("✅ All validation checks passed!")
        print("\n🎉 Preprocessing completed successfully!")
        print("\nNext steps:")
        print("  1. Review visualizations")
        print("  2. Check statistics for anomalies")
        print("  3. Proceed to Phase 2: Model Development")
        return 0
    else:
        print(f"⚠️ Found {len(all_issues)} issues:\n")
        for issue in all_issues:
            print(f"  {issue}")
        
        # Determine severity
        critical = sum(1 for i in all_issues if i.startswith("❌"))
        warnings = sum(1 for i in all_issues if i.startswith("⚠️"))
        
        print(f"\nCritical: {critical}, Warnings: {warnings}")
        
        if critical > 0:
            print("\n❌ Preprocessing incomplete or failed. Please review errors.")
            return 1
        else:
            print("\n⚠️ Preprocessing completed with warnings. Review before proceeding.")
            return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
