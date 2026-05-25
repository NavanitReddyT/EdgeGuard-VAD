import time
import psutil
import json
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class Benchmarker:
    """Tracks and reports performance and accuracy metrics."""
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.cpu_usages = []
        self.ram_usages = []
        self.y_true = []
        self.y_scores = []

    def update(self, y_true=None, y_score=None):
        """Update metrics for each frame."""
        self.frame_count += 1
        self.cpu_usages.append(psutil.cpu_percent())
        self.ram_usages.append(psutil.virtual_memory().used / (1024 * 1024)) # In MB
        if y_true is not None and y_score is not None:
            self.y_true.append(y_true)
            self.y_scores.append(y_score)

    def get_current_fps(self):
        """Calculate the current FPS."""
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0

    def generate_report(self, save_path="results/benchmark_report.json"):
        """Generates and saves a summary of the benchmark."""
        report = {
            "total_frames": self.frame_count,
            "avg_fps": self.get_current_fps(),
            "avg_cpu_usage_percent": sum(self.cpu_usages) / len(self.cpu_usages),
            "avg_ram_usage_mb": sum(self.ram_usages) / len(self.ram_usages),
            "auc_roc": None
        }

        if self.y_true and self.y_scores:
            report["auc_roc"] = roc_auc_score(self.y_true, self.y_scores)
            fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {report["auc_roc"]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig("results/roc_curve.png")

        print("--- Benchmark Report ---")
        for key, value in report.items():
            print(f"{key}: {value}")
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)

        return report
