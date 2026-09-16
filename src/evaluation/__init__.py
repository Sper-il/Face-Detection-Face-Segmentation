"""
Evaluation Metrics Module
"""

from .metrics import evaluate_detection, evaluate_segmentation
from .widerface_eval import evaluate_widerface, load_widerface_annotations
from .fddb_eval import evaluate_fddb, load_fddb_annotations, compute_roc
from .visualization import (
    draw_detection_comparison,
    draw_segmentation_comparison,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    compute_confusion_matrix,
)

__all__ = [
    'evaluate_detection', 'evaluate_segmentation',
    'evaluate_widerface', 'load_widerface_annotations',
    'evaluate_fddb', 'load_fddb_annotations', 'compute_roc',
    'draw_detection_comparison', 'draw_segmentation_comparison',
    'plot_confusion_matrix', 'plot_roc_curve', 'plot_pr_curve',
    'compute_confusion_matrix',
]
