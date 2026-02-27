import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from ewstools import TimeSeries
from ewstools.models import simulate_ricker
from scipy import stats

def seed_torch(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class EWSModelEval(TimeSeries):
    """Extended TimeSeries class for EWS computation"""

    def compute_indicator(self, indicator, **kwargs):
        """Compute specified EWS indicator"""
        rolling_window = kwargs.get("rolling_window") if "rolling_window" in kwargs else 0.5
        if indicator == 'variance':
            self.compute_var(rolling_window=rolling_window)

        elif indicator == 'ac1':
            self.compute_auto(lag=1, rolling_window=rolling_window)
        elif indicator == 'skew':
            self.compute_skew(rolling_window=rolling_window)
        elif indicator == 'kurtosis':
            self.compute_kurt(rolling_window=rolling_window)
        elif indicator == 'cv':
            self.compute_cv(rolling_window=rolling_window)
        else:
            raise ValueError(f"Invalid indicator name: {indicator}")
        self.compute_ktau()
        return self.ews[indicator].dropna().values


def generate_data(data_generate_num=100):
    """Generate transition and null datasets"""
    transition_data = [simulate_ricker(tmax=500, F=[0, 2.7]) for _ in range(data_generate_num)]
    null_data = [simulate_ricker(tmax=500, F=0) for _ in range(data_generate_num)]
    return transition_data, null_data


def create_ews_objects(data, transition, detrend=True):
    """Create and process EWS objects from data"""
    ts = EWSModelEval(data, transition)
    if detrend:
        ts.detrend(method='Gaussian', span=0.2)
    return ts


def calculate_sigma(indicator_values, base_len_ratio=0.3):
    """Calculate sigma values based on baseline"""
    baseline_len = int(base_len_ratio * len(indicator_values))
    baseline = indicator_values[:baseline_len]
    mu = baseline.mean()
    std = baseline.std()
    return (indicator_values - mu) / std


def detect_warning(z_scores, sigma_crit, nr_consecutive: int = 5):
    """Detect warnings using vectorized convolution"""
    z_score = z_scores.flatten()
    warnings = np.where(np.abs(z_score) >= sigma_crit)[0]

    current_seq = []

    for i in warnings:
        if len(current_seq) >= nr_consecutive:
            return True
        elif not current_seq or i == current_seq[-1] + 1:
            current_seq.append(i)
        else:
            current_seq = [i]

    if current_seq and len(current_seq) >= nr_consecutive:

        return True
    else:
        return False


def calculate_ews_roc(transition_series, null_series,
                      thresholds,
                      sigma_max_threshold=150):
    """Calculate ROC/AUC for EWS indicators"""
    null_tau=np.array(null_series)
    trans_tau=np.array(transition_series)


    fpr_list = []  # 假阳性率
    tpr_list = []  # 真阳性率

    # 遍历所有可能的阈值
    for thresh in thresholds:
        # 计算假阳性率（零模型中Kendall系数大于或等于阈值的比例）
        fpr = np.sum(null_tau >= thresh) / len(null_tau)
        fpr_list.append(fpr)

        # 计算真阳性率（过渡模型中Kendall系数大于或等于阈值的比例）
        tpr = np.sum(trans_tau >= thresh) / len(trans_tau)
        tpr_list.append(tpr)

    # 计算AUC值
    roc_auc = auc(fpr_list, tpr_list)

    return fpr_list, tpr_list, roc_auc


def plot_roc_curve(fpr, tpr, auc_value, indicator, save_path):
    """Plot and save ROC curve"""
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=1, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{indicator} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(f"{save_path}/{indicator}_ROC.png")
    plt.close()


def plot_ews_curve(ts, indicator, series_type, save_path):
    """Plot and save EWS time series"""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    # State plot
    ts.state[['state', 'smoothing']].plot(ax=ax1)
    ax1.set_title('System State')
    ts_time = ts.state.index.values
    # EWS indicator plot
    if indicator in ts.ews.columns:

        ts.ews[indicator].plot(ax=ax2, legend=True)
        ax2.set_ylabel(indicator.capitalize())
        ax2.set_title(f'{indicator.capitalize()} Trend')
    else:
        raise ValueError(f"Invalid indicator name: {indicator}")
    plt.xlim(ts_time[0], ts_time[-1])
    plt.tight_layout()

    plt.savefig(f"{save_path}/{series_type}_{indicator}_curve.png")
    plt.close()

def ews_visualize(null_tau, trans_tau, thresholds, fpr, tpr, roc_auc):
    """绘制早期预警信号分析结果"""
    # 1. 创建分布图(a) - 直接计算频率而不是使用高斯拟合
    # 1. 创建分布图(a) - 直接计算频率而不是使用高斯拟合
    step_stype="mid"# 'pre'/'post'/'mid'
    plt.figure(figsize=(14, 10))

    # 创建子图布局
    ax1 = plt.subplot(2, 1, 1)  # 上部分布图
    ax2 = plt.subplot(2, 1, 2)  # 下部ROC曲线图

    # 设置Kendall系数区间为[-1, 1]，细分为20个区间

    bins = thresholds

    # 计算每个区间的频率密度
    null_counts, null_bins = np.histogram(null_tau, bins=bins, density=True)
    trans_counts, trans_bins = np.histogram(trans_tau, bins=bins, density=True)

    # 计算每个区间的中心点
    null_centers = 0.5 * (null_bins[1:] + null_bins[:-1])
    trans_centers = 0.5 * (trans_bins[1:] + trans_bins[:-1])

    # 在分布图(a)上绘制直方图
    ax1.step(null_centers, null_counts, where=step_stype, color='blue',
             label='Null Model (stable system)', linewidth=2)
    ax1.step(trans_centers, trans_counts, where=step_stype, color='red',
             label='Transition Model', linewidth=2)



    # 选择阈值示例,此处选择Youden指数意义下的最佳阈值
    youden_index = np.array(tpr) - np.array(fpr)
    optimal_idx = np.argmax(youden_index)
    threshold=best_threshold = thresholds[optimal_idx]


    # 添加阈值线
    ax1.axvline(x=threshold, color='k', linestyle='--', linewidth=1.5)

    # 填充假阳性区域（零模型中高于阈值部分）
    mask = null_centers >= threshold
    ax1.fill_between(null_centers[mask], 0, null_counts[mask],step=step_stype,
                     color='blue', alpha=0.4, label='False Positive Area')

    # 填充真阳性区域（过渡模型中高于阈值部分）
    mask = trans_centers >= threshold
    ax1.fill_between(trans_centers[mask], 0, trans_counts[mask],step=step_stype,
                     color='red', alpha=0.4, label='True Positive Area')

    # 设置分布图标题和标签
    ax1.set_title('(a) Frequency Distribution of Kendall τ Coefficients', fontsize=16)
    ax1.set_xlabel('Kendall τ Coefficient', fontsize=14)
    ax1.set_ylabel('Frequency Density', fontsize=14)
    ax1.set_xlim([-1.0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # 2. 绘制ROC曲线
    ax2.plot(fpr, tpr, 'k-', linewidth=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)  # 随机分类线

    # 标记示例阈值点，
    thresh_idx = np.argmin(np.abs(thresholds - threshold))
    ax2.plot(fpr[thresh_idx], tpr[thresh_idx], 'bo', markersize=8,
             label=f'current threshold')
    # 选择最佳阈值点并标记
    ax2.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
             label=f'Youden Index = {youden_index[optimal_idx]:.2f}\n best Threshold = {best_threshold:.2f}')



    # 设置ROC曲线标题和标签
    ax2.set_title('(b) ROC Curve', fontsize=16)
    ax2.set_xlabel('False Positive Rate', fontsize=14)
    ax2.set_ylabel('True Positive Rate', fontsize=14)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 添加互动功能：通过滑块动态调整阈值
    from matplotlib.widgets import Slider

    axcolor = 'lightgoldenrodyellow'
    axthresh = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor=axcolor)
    sthresh = Slider(axthresh, 'Threshold', -1.0, 1.0, valinit=threshold)


    def update(val):
        """更新图表以响应阈值变化"""
        # 获取当前阈值
        thresh_val = sthresh.val

        # 更新分布图中的阈值线
        ax1.lines[2].set_xdata([thresh_val, thresh_val])

        # 更新填充区域
        for coll in ax1.collections[:]:
            coll.remove()

        # 填充假阳性区域（零模型中高于阈值部分）
        mask = null_centers >= thresh_val
        ax1.fill_between(null_centers[mask], 0, null_counts[mask],step=step_stype,
                         color='blue', alpha=0.4)

        # 填充真阳性区域（过渡模型中高于阈值部分）
        mask = trans_centers >= thresh_val
        ax1.fill_between(trans_centers[mask], 0, trans_counts[mask],step=step_stype,
                         color='red', alpha=0.4)

        # 更新ROC曲线中的标记点
        thresh_idx = np.argmin(np.abs(thresholds - thresh_val))
        ax2.lines[2].set_data([fpr[thresh_idx]], [tpr[thresh_idx]])

        # 重绘图表
        plt.draw()


    # 将更新函数与滑块绑定
    sthresh.on_changed(update)

    plt.tight_layout()
    plt.savefig('EWS_ROC_Analysis.png', dpi=300)
    plt.show()
def evaluate_ews(transition_data, null_data, ews_indicators,
                 save_path, seed=123, transition=440,):
    """Main function to evaluate EWS indicators"""
    seed_torch(seed)

    # Create TimeSeries objects in vectorized manner
    ts_trans_list = [create_ews_objects(data, transition) for data in transition_data]
    ts_null_list = [create_ews_objects(data, transition) for data in null_data]

    results = {}

    for indicator in ews_indicators:
        print(f"Evaluating {indicator}...")
        # Compute indicators for all series
        time_start = time.time()
        trans_ews = [ts.compute_indicator(indicator) for ts in ts_trans_list]
        null_ews = [ts.compute_indicator(indicator) for ts in ts_null_list]
        print("cost time:", time.time() - time_start)
        # Calculate kendall tau correlation coefficient

        trans_tau = [ts.ktau[indicator] for ts in ts_trans_list]
        null_tau = [ts.ktau[indicator] for ts in ts_null_list]




        # Compute ROC/AUC
        #  time_start=time.time()
        thresholds = np.linspace(-1, 1, 100)
        fpr, tpr, auc_value = calculate_ews_roc(trans_tau, null_tau,thresholds)
        # results[indicator] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_value}


        # Generate plots
        plot_roc_curve(fpr, tpr, auc_value, indicator, save_path)
        plot_ews_curve(ts_trans_list[0], indicator, "transition", save_path)
        plot_ews_curve(ts_null_list[0], indicator, "null", save_path)

        ews_visualize(null_tau, trans_tau, thresholds, fpr, tpr, auc_value)

        print(f"{indicator}: AUC = {auc_value:.4f}")

    return results


if __name__ == "__main__":
    # Constants
    DATA_GENERATE_NUM = 250

    SAVE_PATH = "ews_AUC_kendell_figs"
    EWS_INDICATORS = ['variance', 'ac1', 'skew', 'kurtosis', 'cv']
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    transition_data, null_data = generate_data(DATA_GENERATE_NUM)
    ews_results = evaluate_ews(transition_data, null_data,
                               ews_indicators=EWS_INDICATORS,
                               save_path=SAVE_PATH,
                               )