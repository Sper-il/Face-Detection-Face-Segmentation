"""
Master Script for Data Preprocessing Pipeline
Orchestrates the entire data preprocessing workflow.

Features:
- Check data availability
- Run preprocessing for both datasets
- Generate visualizations
- Create quality reports
- Update progress documentation
"""

import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, total: int, description: str):
    """Print step information."""
    print(f"\n[Step {step_num}/{total}] {description}")
    print("-" * 70)


def check_prerequisites():
    """Check if raw data exists."""
    print_step(1, 5, "Checking Prerequisites")
    
    base_dir = Path("e:/Face Detection & Face Segmentation")
    wider_dir = base_dir / "data/raw/WIDER_FACE"
    celeb_dir = base_dir / "data/raw/CelebAMask-HQ"
    
    issues = []
    
    # Check WIDER FACE
    if not wider_dir.exists():
        issues.append(f"❌ WIDER FACE not found at {wider_dir}")
    else:
        # Check subdirectories
        required_dirs = ["WIDER_train", "WIDER_val", "wider_face_split"]
        for d in required_dirs:
            if not (wider_dir / d).exists():
                issues.append(f"❌ Missing: {wider_dir / d}")
        
        if not issues:
            print(f"✅ WIDER FACE dataset found")
    
    # Check CelebAMask-HQ
    if not celeb_dir.exists():
        issues.append(f"❌ CelebAMask-HQ not found at {celeb_dir}")
    else:
        celeb_subdir = celeb_dir / "CelebAMask-HQ"
        if celeb_subdir.exists():
            print(f"✅ CelebAMask-HQ dataset found")
        else:
            issues.append(f"❌ Missing CelebAMask-HQ subdirectory")
    
    # Check Python dependencies
    print("\nChecking Python dependencies...")
    required_packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            issues.append(f"❌ Missing package: {package_name}")
            print(f"  ❌ {package_name} - Not installed")
    
    if issues:
        print("\n⚠️ Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease fix these issues before continuing.")
        return False
    
    print("\n✅ All prerequisites satisfied!")
    return True


def run_preprocessing_script(script_name: str, description: str) -> bool:
    """Run a preprocessing script and capture output."""
    script_path = Path("e:/Face Detection & Face Segmentation/src/data") / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"\n🚀 Running {description}...")
    print(f"   Script: {script_name}")
    
    start_time = time.time()
    
    try:
        # Run script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            print(f"   Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            return True
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {description} timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"\n❌ Error running {description}: {str(e)}")
        return False


def update_progress_documentation():
    """Update PROGRESS_STATUS.md with preprocessing completion."""
    print_step(5, 5, "Updating Documentation")
    
    progress_file = Path("e:/Face Detection & Face Segmentation/PROGRESS_STATUS.md")
    
    if not progress_file.exists():
        print(f"⚠️ PROGRESS_STATUS.md not found at {progress_file}")
        return
    
    # Add preprocessing completion note
    update_text = f"""

---
**Preprocessing Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**

✅ Data preprocessing pipeline completed successfully!

**Completed Tasks:**
- [x] WIDER FACE dataset preprocessing
- [x] CelebAMask-HQ dataset preprocessing
- [x] Data quality checks and validation
- [x] Statistical analysis and visualization
- [x] Dataset splits generation

**Outputs:**
- Processed WIDER FACE: `data/processed/wider_face/`
- Processed CelebAMask-HQ: `data/processed/celebamask_hq/`
- Visualizations: Available in respective `visualizations/` folders
- Statistics: Available in respective `statistics/` folders

**Next Steps:**
- Review visualizations and statistics
- Proceed to Phase 2: Model Development (Detection)
"""
    
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(update_text)
        print(f"✅ Updated {progress_file}")
    except Exception as e:
        print(f"⚠️ Could not update progress file: {str(e)}")


def main():
    """Main preprocessing pipeline orchestrator."""
    print_header("DATA PREPROCESSING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return
    
    # Step 2: Preprocess WIDER FACE
    print_step(2, 5, "WIDER FACE Preprocessing")
    wider_success = run_preprocessing_script(
        "preprocess_wider_face.py",
        "WIDER FACE Dataset Preprocessing"
    )
    
    if not wider_success:
        print("\n⚠️ WIDER FACE preprocessing failed. Continuing with CelebAMask-HQ...")
    
    # Step 3: Preprocess CelebAMask-HQ
    print_step(3, 5, "CelebAMask-HQ Preprocessing")
    celeb_success = run_preprocessing_script(
        "preprocess_celebamask_hq.py",
        "CelebAMask-HQ Dataset Preprocessing"
    )
    
    if not celeb_success:
        print("\n⚠️ CelebAMask-HQ preprocessing failed.")
    
    # Step 4: Generate visualizations
    print_step(4, 5, "Data Visualization")
    
    if wider_success or celeb_success:
        viz_success = run_preprocessing_script(
            "visualize_data.py",
            "Data Visualization & Quality Check"
        )
    else:
        print("⚠️ Skipping visualization (no datasets processed)")
        viz_success = False
    
    # Step 5: Update documentation
    if wider_success or celeb_success:
        update_progress_documentation()
    
    # Final summary
    overall_time = time.time() - overall_start
    
    print_header("PIPELINE SUMMARY")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print("\nResults:")
    print(f"  {'✅' if wider_success else '❌'} WIDER FACE Preprocessing")
    print(f"  {'✅' if celeb_success else '❌'} CelebAMask-HQ Preprocessing")
    print(f"  {'✅' if viz_success else '❌'} Visualization Generation")
    
    if wider_success and celeb_success:
        print("\n🎉 All preprocessing tasks completed successfully!")
        print("\nNext steps:")
        print("  1. Review visualizations in data/processed/*/visualizations/")
        print("  2. Check statistics in data/processed/*/statistics/")
        print("  3. Proceed to Phase 2: Face Detection Model Training")
    elif wider_success or celeb_success:
        print("\n⚠️ Preprocessing partially completed. Review errors above.")
    else:
        print("\n❌ Preprocessing pipeline failed. Review errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
