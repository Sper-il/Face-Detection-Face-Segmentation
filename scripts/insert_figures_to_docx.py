"""
Insert Figures and Tables into BAO_CAO.docx
============================================
This script inserts all generated figures and tables from docs/figures/
into the BAO_CAO.docx report at the correct positions.

Usage:
    python scripts/insert_figures_to_docx.py [--preview]
"""

import re
import shutil
import argparse
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Optional

try:
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn, nsmap
    from docx.oxml import OxmlElement
except ImportError:
    print("ERROR: python-docx not installed. Run: pip install python-docx")
    import sys
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
DOCX_PATH = PROJECT_ROOT / "BAO_CAO.docx"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


# ============================================================================
# FIGURE & TABLE DEFINITIONS
# ============================================================================

# Order: (figure_id, caption_id, image_filename, alt_text, search_keywords)
FIGURES_TO_INSERT = [
    {
        "id": "1.1",
        "caption": "Hình 1.1",
        "image": "fig_1_1_pipeline_overview.png",
        "alt": "Sơ đồ tổng quan pipeline 2-stage",
        "after_text": "1.1.2 Đề tài của đồ án",  # Insert after this section
        "width_inches": 6.5,
    },
    {
        "id": "2.1",
        "caption": "Bảng 2.1",
        "image": "fig_2_1_detection_comparison.png",
        "alt": "So sánh các mô hình face detection",
        "after_text": "Kết luận: RetinaFace là lựa chọn",
        "width_inches": 6.5,
    },
    {
        "id": "2.2",
        "caption": "Bảng 2.2",
        "image": "fig_2_2_segmentation_comparison.png",
        "alt": "So sánh các mô hình face segmentation",
        "after_text": "Mask R-CNN (2017)",
        "width_inches": 6.5,
    },
    {
        "id": "3.1",
        "caption": "Hình 3.1",
        "image": "fig_3_1_wider_face_distribution.png",
        "alt": "Phân bố dữ liệu WIDER FACE theo 61 event",
        "after_text": "Pose variation",
        "width_inches": 6.5,
    },
    {
        "id": "3.2",
        "caption": "Hình 3.2",
        "image": "fig_3_2_celebamask_classes.png",
        "alt": "Mẫu ảnh CelebAMask-HQ với 19 lớp semantic",
        "after_text": "3.3.3 Thống kê dữ liệu",
        "width_inches": 6.5,
    },
    {
        "id": "4.1",
        "caption": "Hình 4.1",
        "image": "fig_4_1_pipeline_architecture.png",
        "alt": "Kiến trúc pipeline 2-stage",
        "after_text": "4.1 Kiến trúc tổng thể",
        "width_inches": 6.5,
    },
    {
        "id": "4.2",
        "caption": "Hình 4.2",
        "image": "fig_4_2_retinaface_architecture.png",
        "alt": "Sơ đồ khối RetinaFace",
        "after_text": "4.3 Module phân đoạn",
        "width_inches": 6.5,
    },
    {
        "id": "4.3",
        "caption": "Hình 4.3",
        "image": "fig_4_3_unet_architecture.png",
        "alt": "Kiến trúc U-Net",
        "after_text": "4.3.5 Skip Connections",
        "width_inches": 6.5,
    },
    {
        "id": "4.4",
        "caption": "Hình 4.4",
        "image": "fig_4_4_skip_connections.png",
        "alt": "Skip connection trong U-Net",
        "after_text": "4.3.6 Output Layer",
        "width_inches": 6.5,
    },
    {
        "id": "5.1",
        "caption": "Hình 5.1",
        "image": "fig_5_1_training_loss_curves.png",
        "alt": "Đường cong loss U-Net trong quá trình huấn luyện",
        "after_text": "5.4 Kết quả quá trình huấn luyện",
        "width_inches": 6.5,
    },
    {
        "id": "6.1",
        "caption": "Hình 6.1",
        "image": "fig_6_1_iou_dice_curves.png",
        "alt": "Đường cong IoU và Dice trên validation set",
        "after_text": "6.3 Phân tích lỗi",
        "width_inches": 6.5,
    },
    {
        "id": "6.2",
        "caption": "Hình 6.2",
        "image": "fig_6_2_segmentation_visualization.png",
        "alt": "Visualization: Ảnh gốc - Mask dự đoán - Overlay",
        "after_text": "6.4 Trực quan hóa kết quả",
        "width_inches": 6.5,
    },
    {
        "id": "7.1",
        "caption": "Hình 7.1",
        "image": "fig_7_1_pipeline_demo.png",
        "alt": "Demo kết quả end-to-end pipeline",
        "after_text": "7.3 Triển khai và xuất mô hình",
        "width_inches": 6.5,
    },
]


# Table data definitions
TABLES_TO_INSERT = [
    {
        "id": "3.1",
        "title": "Bảng 3.1: Thống kê WIDER FACE Dataset",
        "after_text": "Pose variation",
        "headers": ["Thông số", "Giá trị"],
        "rows": [
            ["Tổng số ảnh", "32,203"],
            ["Tổng số khuôn mặt", "~393,000"],
            ["Số event (categories)", "61"],
            ["Tập Train", "12,880 ảnh"],
            ["Tập Validation", "3,226 ảnh"],
            ["Tập Test", "16,097 ảnh"],
            ["Resolution đa dạng", "1024x768 trung bình"],
            ["Format annotation", "Bounding box + 5 landmarks"],
            ["Pose variation", "Có (frontal, profile, occlusion)"],
            ["Illumination variation", "Có"],
            ["Occlusion level", "Easy / Medium / Hard"],
        ],
    },
    {
        "id": "3.2",
        "title": "Bảng 3.2: Thống kê CelebAMask-HQ Dataset",
        "after_text": "3.3.3 Thống kê dữ liệu",
        "headers": ["Thông số", "Giá trị"],
        "rows": [
            ["Tổng số ảnh", "30,000"],
            ["Tập Train", "24,000 ảnh (80%)"],
            ["Tập Validation", "3,000 ảnh (10%)"],
            ["Tập Test", "3,000 ảnh (10%)"],
            ["Resolution gốc", "512 × 512 pixels"],
            ["Resolution sau preprocess", "256 × 256 pixels"],
            ["Số lớp semantic", "19 lớp"],
            ["Format annotation", "Pixel-level masks"],
            ["Source", "CelebA-HQ"],
            ["Nhiệm vụ trong đồ án", "Binary face mask (background + face)"],
        ],
    },
    {
        "id": "4.1",
        "title": "Bảng 4.1: Cấu hình U-Net chi tiết",
        "after_text": "4.3.7 Bảng thông số U-Net",
        "headers": ["Layer", "Input Size", "Output Channels", "Parameters"],
        "rows": [
            ["Input", "256×256×3", "-", "-"],
            ["Enc1", "256×256×3", "64", "~1,792"],
            ["Enc2", "128×128×64", "128", "~73,728"],
            ["Enc3", "64×64×128", "256", "~294,912"],
            ["Enc4", "32×32×256", "512", "~1,179,648"],
            ["Bottleneck", "16×16×512", "1024", "~4,718,592"],
            ["Dec4", "32×32×1024", "512", "~4,718,592"],
            ["Dec3", "64×64×512", "256", "~1,179,648"],
            ["Dec2", "128×128×256", "128", "~294,912"],
            ["Dec1", "256×256×128", "64", "~73,728"],
            ["Output", "256×256×64", "2", "~130"],
            ["Tổng cộng", "-", "-", "31,043,586"],
        ],
    },
    {
        "id": "5.1",
        "title": "Bảng 5.1: Hyperparameters huấn luyện U-Net",
        "after_text": "5.4 Kết quả quá trình huấn luyện",
        "headers": ["Hyperparameter", "Detection", "Segmentation"],
        "rows": [
            ["Model", "RetinaFace", "U-Net"],
            ["Input Size", "640×640", "256×256"],
            ["Batch Size", "8", "16"],
            ["Epochs", "12", "10"],
            ["Learning Rate", "1e-4", "1e-3"],
            ["Optimizer", "AdamW", "Adam"],
            ["Weight Decay", "1e-4", "1e-4"],
            ["LR Scheduler", "CosineAnnealingLR", "CosineAnnealingLR"],
            ["Loss Function", "Multi-task (cls + box + lm)", "CE + Dice"],
            ["Loss Weights", "1:1:1", "0.5:0.5"],
            ["Hardware", "Kaggle T4 (16GB)", "Kaggle T4 (16GB)"],
            ["Training Time", "Pending", "~3 hours"],
        ],
    },
    {
        "id": "6.1",
        "title": "Bảng 6.1: Kết quả segmentation trên test set",
        "after_text": "6.2 Kết quả phân đoạn",
        "headers": ["Metric", "Value", "Target", "Status"],
        "rows": [
            ["Mean IoU", "0.9682", ">= 0.90", "PASS"],
            ["Mean Dice", "0.9835", ">= 0.95", "PASS"],
            ["Pixel Accuracy", "0.9769", ">= 0.97", "PASS"],
            ["Precision (face)", "0.9827", "-", "-"],
            ["Recall (face)", "0.9849", "-", "-"],
            ["F1-Score (face)", "0.9835", "-", "-"],
        ],
    },
    {
        "id": "6.2",
        "title": "Bảng 6.2: Kết quả segmentation trên val set",
        "after_text": "6.2 Kết quả phân đoạn",
        "headers": ["Split", "# Images", "IoU", "Dice", "Pixel Acc"],
        "rows": [
            ["Validation", "3,000", "0.9633", "0.9807", "0.9728"],
            ["Test", "3,000", "0.9682", "0.9835", "0.9769"],
            ["Mean", "-", "0.9658", "0.9821", "0.9749"],
        ],
    },
    {
        "id": "7.1",
        "title": "Bảng 7.1: Thời gian inference U-Net",
        "after_text": "7.3 Triển khai",
        "headers": ["Input Size", "Device", "Time (ms)", "FPS"],
        "rows": [
            ["256×256", "CPU (Intel i7)", "~1100", "~0.9"],
            ["256×256", "GPU (T4)", "~25", "~40"],
            ["256×256", "GPU (V100)", "~12", "~83"],
            ["512×512", "GPU (T4)", "~100", "~10"],
            ["512×512", "GPU (V100)", "~48", "~21"],
            ["Batch=4, 256×256", "GPU (T4)", "~80", "~50"],
            ["Batch=8, 256×256", "GPU (T4)", "~150", "~53"],
        ],
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_caption_after_paragraph(doc, target_paragraph, caption_text, image_path, 
                                 width_inches=6.5, alt_text=""):
    """Add a figure (image + caption) after a target paragraph."""
    if not image_path.exists():
        print(f"  [WARN] Image not found: {image_path}")
        return False

    # Create a new paragraph for the image
    img_paragraph = target_paragraph.insert_paragraph_before("")
    img_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add image
    run = img_paragraph.add_run()
    try:
        run.add_picture(str(image_path), width=Inches(width_inches))
    except Exception as e:
        print(f"  [ERROR] Failed to add image: {e}")
        return False

    # Add caption paragraph
    caption_paragraph = target_paragraph.insert_paragraph_before("")
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_paragraph.add_run(caption_text)
    caption_run.italic = True
    caption_run.font.size = Pt(10)
    caption_run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    return True


def add_table_after_paragraph(doc, target_paragraph, table_data):
    """Add a Word table after the target paragraph."""
    title = table_data['title']
    headers = table_data['headers']
    rows = table_data['rows']
    n_cols = len(headers)

    # Move target paragraph down (we'll insert before it)
    # First, create the title paragraph
    title_para = target_paragraph.insert_paragraph_before("")
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    title_run.bold = True
    title_run.font.size = Pt(11)

    # Create table
    table_para = target_paragraph.insert_paragraph_before("")
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)

    # Move table into our paragraph
    table._element.getparent().remove(table._element)
    table_para._p.addnext(table._element)

    # Apply table style
    try:
        table.style = 'Light Grid Accent 1'
    except KeyError:
        try:
            table.style = 'Table Grid'
        except KeyError:
            table.style = 'Normal Table'

    # Header row
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        for paragraph in hdr_cells[i].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data rows
    for row_idx, row_data in enumerate(rows, start=1):
        row_cells = table.rows[row_idx].cells
        for col_idx, cell_data in enumerate(row_data):
            row_cells[col_idx].text = str(cell_data)
            for paragraph in row_cells[col_idx].paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def find_paragraph_containing(doc, search_text: str, occurrence: int = 0):
    """Find a paragraph containing the search text."""
    matches = []
    for i, para in enumerate(doc.paragraphs):
        if search_text in para.text:
            matches.append(para)
    if occurrence < len(matches):
        return matches[occurrence]
    return None


def insert_all_figures(doc, FIGURES_DIR, dry_run=False):
    """Insert all figures into the document."""
    print("\n" + "=" * 60)
    print("INSERTING FIGURES")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    skipped_count = 0

    for fig_info in FIGURES_TO_INSERT:
        fig_id = fig_info['id']
        caption = fig_info['caption']
        image_name = fig_info['image']
        after_text = fig_info['after_text']
        width = fig_info['width_inches']

        image_path = FIGURES_DIR / image_name

        print(f"\n[{fig_id}] {fig_info['alt']}")
        print(f"  Image: {image_name}")

        if not image_path.exists():
            print(f"  [SKIP] Image not found")
            skipped_count += 1
            continue

        # Find target paragraph
        target = find_paragraph_containing(doc, after_text)
        if not target:
            print(f"  [FAIL] Could not find anchor text: {after_text}")
            fail_count += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would insert after: {after_text[:50]}...")
            continue

        # Insert figure and caption
        try:
            success = add_caption_after_paragraph(
                doc, target, caption, image_path,
                width_inches=width, alt_text=fig_info['alt']
            )
            if success:
                print(f"  [OK] Inserted with caption '{caption}'")
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            fail_count += 1

    print(f"\nFigures Summary: {success_count} success, {fail_count} failed, {skipped_count} skipped")


def insert_all_tables(doc, dry_run=False):
    """Insert all tables into the document."""
    print("\n" + "=" * 60)
    print("INSERTING TABLES")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for table_info in TABLES_TO_INSERT:
        tbl_id = table_info['id']
        title = table_info['title']
        after_text = table_info['after_text']

        print(f"\n[{tbl_id}] {title}")

        # Find target paragraph
        target = find_paragraph_containing(doc, after_text)
        if not target:
            print(f"  [FAIL] Could not find anchor text: {after_text}")
            fail_count += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would insert after: {after_text[:50]}...")
            continue

        # Insert table
        try:
            add_table_after_paragraph(doc, target, table_info)
            print(f"  [OK] Table {tbl_id} inserted")
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            fail_count += 1

    print(f"\nTables Summary: {success_count} success, {fail_count} failed")


def update_list_of_figures_and_tables(doc):
    """Update the DANH MỤC HÌNH VẼ and DANH MỤC BẢNG BIỂU sections."""
    print("\n" + "=" * 60)
    print("UPDATING LIST OF FIGURES AND TABLES")
    print("=" * 60)
    print("  [NOTE] The list of figures/tables is already in the document.")
    print("          Open in Word and press F9 to update fields if needed.")


def main():
    parser = argparse.ArgumentParser(description='Insert figures and tables into BAO_CAO.docx')
    parser.add_argument('--preview', '--dry-run', action='store_true',
                       help='Preview what would be inserted without modifying the file')
    parser.add_argument('--backup', action='store_true',
                       help='Create a backup of the original file')
    args = parser.parse_args()

    if not DOCX_PATH.exists():
        print(f"ERROR: {DOCX_PATH} not found")
        return 1

    if args.backup:
        backup_path = DOCX_PATH.with_suffix('.docx.bak')
        shutil.copy2(DOCX_PATH, backup_path)
        print(f"[INFO] Backup created: {backup_path}")

    print(f"Loading document: {DOCX_PATH}")
    doc = Document(str(DOCX_PATH))

    print(f"Total paragraphs: {len(doc.paragraphs)}")
    print(f"Total tables: {len(doc.tables)}")

    if args.preview:
        print("\n[PREVIEW MODE] No changes will be made.")
        insert_all_figures(doc, FIGURES_DIR, dry_run=True)
        insert_all_tables(doc, dry_run=True)
        return 0

    # Insert figures and tables
    insert_all_figures(doc, FIGURES_DIR, dry_run=False)
    insert_all_tables(doc, dry_run=False)

    # Update lists
    update_list_of_figures_and_tables(doc)

    # Save
    print("\n" + "=" * 60)
    print("SAVING DOCUMENT")
    print("=" * 60)

    output_path = DOCX_PATH.with_name('BAO_CAO_with_figures.docx')
    doc.save(str(output_path))
    print(f"\n[OK] Saved to: {output_path}")
    print(f"\nFinal Statistics:")
    print(f"  - Total paragraphs: {len(doc.paragraphs)}")
    print(f"  - Total tables: {len(doc.tables)}")
    print(f"  - File size: {output_path.stat().st_size:,} bytes")

    return 0


if __name__ == '__main__':
    exit(main())
