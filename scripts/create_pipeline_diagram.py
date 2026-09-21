import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False, text_color='#1a1a1a'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor=color, edgecolor='#444444', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, label='', label_x=None, label_y=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=2))
    if label:
        lx = label_x if label_x else (x1 + x2) / 2
        ly = label_y if label_y else (y1 + y2) / 2 + 0.15
        ax.text(lx, ly, label, ha='center', va='center', fontsize=8, color='#333333',
                style='italic', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'))

# Title
ax.text(7, 9.5, "Face Detection & Face Segmentation Pipeline", ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1A237E')

# ===== Stage 1 area =====
ax.add_patch(FancyBboxPatch((0.5, 0.8), 6.3, 7.6, boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2.5))
ax.text(3.65, 7.85, "STAGE 1: Face Detection (RetinaFace)", ha='center', fontsize=13,
        fontweight='bold', color='#1565C0', backgroundcolor='white')

# Input
draw_box(ax, 0.7, 4.0, 1.5, 1.6, 'INPUT\n\nImage\n(512×512)', '#E8F5E9', 10, True)

# Stage 1 modules
draw_box(ax, 2.8, 4.2, 1.4, 1.4, 'Backbone\n(ResNet-34)', '#BBDEFB', 9, True)
draw_box(ax, 4.5, 4.2, 1.4, 1.4, 'FPN\nFeature\nPyramid', '#BBDEFB', 9, True)
draw_box(ax, 6.2, 4.2, 0.5, 1.4, '', '#BBDEFB')

# ===== Stage 2 area =====
ax.add_patch(FancyBboxPatch((7.2, 0.8), 6.3, 7.6, boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2.5))
ax.text(10.35, 7.85, "STAGE 2: Face Segmentation (U-Net)", ha='center', fontsize=13,
        fontweight='bold', color='#2E7D32', backgroundcolor='white')

# Stage 2 modules
draw_box(ax, 7.4, 4.2, 1.5, 1.4, 'Encoder\n(Conv blocks\n+ MaxPool)', '#C8E6C9', 9, True)
draw_box(ax, 9.2, 4.2, 1.4, 1.4, 'Bottleneck', '#C8E6C9', 9, True)
draw_box(ax, 10.9, 4.2, 1.6, 1.4, 'Decoder\n(UpConv\n+ Skip)', '#C8E6C9', 9, True)

# ===== Arrows in Stage 1 =====
draw_arrow(ax, 2.2, 5.0, 2.8, 5.0)
draw_arrow(ax, 4.2, 5.0, 4.5, 5.0)
draw_arrow(ax, 5.9, 5.0, 6.2, 5.0)

# ===== Arrows in Stage 2 =====
draw_arrow(ax, 8.9, 5.0, 9.2, 5.0)
draw_arrow(ax, 10.6, 5.0, 10.9, 5.0)

# ===== Stage 1 -> Stage 2 =====
draw_arrow(ax, 6.7, 4.0, 6.7, 2.2, 'Face crop\n+ BBox', label_x=6.7, label_y=3.1)

# ===== Stage 2 input from image =====
draw_arrow(ax, 2.2, 4.0, 2.2, 2.5)
draw_arrow(ax, 2.2, 2.5, 7.4, 2.5, 'Face\ncrop', label_x=5.0, label_y=2.35)

# ===== Intermediate output (BBox) =====
draw_box(ax, 5.5, 1.5, 2.2, 1.1, 'Face BBox\n+ Landmarks', '#FFF59D', 9, True)

# ===== Segmentation output =====
draw_box(ax, 10.5, 1.5, 2.0, 1.1, 'Segmentation\nMask', '#C8E6C9', 9, True)
draw_arrow(ax, 11.7, 4.2, 11.7, 2.6)

# ===== Final output =====
draw_box(ax, 6.5, 0.15, 5.5, 0.9, 'FINAL OUTPUT: BBox + Face Segmentation Mask',
         '#F3E5F5', 10, True)
ax.annotate('', xy=(6.9, 1.5), xytext=(6.9, 1.5),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.annotate('', xy=(11.1, 1.5), xytext=(11.1, 1.5),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.annotate('', xy=(8.0, 0.95), xytext=(7.1, 1.5), arrowprops=dict(arrowstyle='<-', color='#555555', lw=1.5))
ax.annotate('', xy=(10.5, 0.95), xytext=(11.0, 1.5), arrowprops=dict(arrowstyle='<-', color='#555555', lw=1.5))

# ===== Bottom annotations =====
ax.text(3.65, 0.15, '↓ Face Detection', ha='center', fontsize=9, color='#1565C0')
ax.text(10.35, 0.15, '↓ Segmentation', ha='center', fontsize=9, color='#2E7D32')

plt.tight_layout()
plt.savefig('e:/Face-Detection-Face-Segmentation/docs/figures/fig_4_1_pipeline_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig_4_1_pipeline_architecture.png")
