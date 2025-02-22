import datetime
import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from MyLogger import logger

def save_report_and_confusion_matrix(y_true, y_pred, labels, output_dir="results"):
    """
    保存混淆矩阵和分类报告到指定目录。
    Args:
        y_true (list): 真实标签。
        y_pred (list): 预测标签。
        labels (list): 类别标签。
        output_dir (str): 保存结果的目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    short_labels = [label for label in labels]

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix_{current_time}.png")
    plt.savefig(cm_path)
    plt.close()

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_path = os.path.join(output_dir, f"classification_report_{current_time}.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=labels))
    logger.info(f"Confusion matrix saved to {cm_path}")
    logger.info(f"Classification report saved to {report_path}")
