import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from ewstools import TimeSeries
from ewstools.models import simulate_ricker





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
        return self.ews[indicator].dropna().values


def generate_data(data_generate_num=100):
    """Generate transition and null datasets"""
    transition_data = [simulate_ricker(tmax=500, F=[0, 2.7]) for _ in range(data_generate_num)]
    null_data = [simulate_ricker(tmax=500, F=0) for _ in range(data_generate_num)]
    return transition_data, null_data


def create_ews_objects(data, transition,detrend=True):
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
    return  (indicator_values - mu) / std


def detect_warning(z_scores, sigma_crit, nr_consecutive:int =5):
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



def get_first_warning( z_scores, sigma_crit,nr_consecutive:int =5):
    # 找出超出阈值的点
    z_scores = z_scores.flatten()
    pre_ews_len = len(z_scores)

    warnings_mask = np.abs(z_scores) >= sigma_crit
   # print("warnings_mask shape:",warnings_mask)
    warning_idxs = np.where(warnings_mask)[0]

    current_seq = []
    # 找出第一个连续警报的预警点
    for i in warning_idxs:
        if len(current_seq) >= nr_consecutive:
            break
        elif not current_seq or i == current_seq[-1] + 1:
            current_seq.append(i)
        else:
            current_seq = [i]
    if current_seq and len(current_seq) >= nr_consecutive:
        first_idx = current_seq[0]-pre_ews_len+1
    else:
        first_idx = 0

    # 确定方向（正 true/负 false）
    direction = True if z_scores[pre_ews_len-1+first_idx] > 0 else False
    return first_idx, direction
def get_first_warning_statistic(z_scores, sigma_crit,nr_consecutive:int =5):
    timing_ews=[]
    direction_ews=[]
    for z_score in z_scores:

        first_idx, direction = get_first_warning(z_score, sigma_crit,nr_consecutive)
        timing_ews.append(first_idx)
        direction_ews.append(direction)

    timing_ews_median=np.median(np.array(timing_ews))
    #100% of positive warning signs means that all warnings showed anincrease
    # 0% means that all showed a decreas
    # 50% means an equal mix of increases and decreases
    direction_per=sum(direction_ews)/len(direction_ews)#100%代表

    return timing_ews_median,direction_per
def calculate_ews_roc(transition_series, null_series,
                      sigma_crit_num=1000,
                      ):
    """Calculate ROC/AUC for EWS indicators"""
    max_sigma = max(np.max(np.abs(transition_series)), np.max(np.abs(null_series)))
    sigma_crit_step=max_sigma/sigma_crit_num
    sigma_crits = np.arange(0, max_sigma + sigma_crit_step, sigma_crit_step)

    tpr_list, fpr_list = [], []

    for sc in sigma_crits:
        tp_count = np.sum([detect_warning(ts, sc) for ts in transition_series])
        fp_count = np.sum([detect_warning(ts, sc) for ts in null_series])

        tpr_list.append(tp_count / len(transition_series))
        fpr_list.append(fp_count / len(null_series))

    # Reverse for ROC curve plotting
    tpr_list.reverse()
    fpr_list.reverse()
    sigma_crits=np.flip(sigma_crits)

    return fpr_list, tpr_list, auc(fpr_list, tpr_list),sigma_crits


def plot_roc_curve(fpr, tpr, auc_value,thresholds, indicator, save_path):
    """Plot and save ROC curve"""
    youden_index = np.array(tpr) - np.array(fpr)
    optimal_idx = np.argmax(youden_index)
    best_threshold = thresholds[optimal_idx]
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=1, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
             label=f'Youden Index = {youden_index[optimal_idx]:.2f}\n best Threshold = {best_threshold:.2f}')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
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
    ts_time=ts.state.index.values
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


def evaluate_ews(transition_data, null_data, ews_indicators,
                 save_path,seed=123,transition=440,base_len_ratio=0.3):
    """Main function to evaluate EWS indicators"""
    seed_torch(seed)


    # Create TimeSeries objects in vectorized manner
    ts_trans_list = [create_ews_objects(data, transition) for data in transition_data]
    ts_null_list = [create_ews_objects(data, transition) for data in null_data]

    results = {}

    for indicator in ews_indicators:
        print(f"Evaluating {indicator}...")
        # Compute indicators for all series
        trans_ews = [ts.compute_indicator(indicator) for ts in ts_trans_list]
        null_ews = [ts.compute_indicator(indicator) for ts in ts_null_list]

        # Calculate sigma values
        trans_sigma = [calculate_sigma(ews,base_len_ratio=base_len_ratio) for ews in trans_ews]
        null_sigma = [calculate_sigma(ews,base_len_ratio=base_len_ratio) for ews in null_ews]

        # Compute ROC/AUC
      #  time_start=time.time()
        fpr, tpr, auc_value,thresholds = calculate_ews_roc(trans_sigma, null_sigma)
        # results[indicator] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_value}
        # print("cost time:",time.time()-time_start)
        youden_index = np.array(tpr) - np.array(fpr)
        optimal_idx = np.argmax(youden_index)
        best_threshold = thresholds[optimal_idx]
        timing, direction=get_first_warning_statistic(trans_sigma, best_threshold)
        # Generate plots
        plot_roc_curve(fpr, tpr, auc_value,thresholds, indicator,save_path)
        plot_ews_curve(ts_trans_list[0], indicator, "transition",save_path)
        plot_ews_curve(ts_null_list[0], indicator, "null",save_path)

        print(f"{indicator}: AUC = {auc_value:.4f}","best threshold:",best_threshold)
        print(f"timing:",timing,"direction:",direction)

    return results


if __name__ == "__main__":
    # Constants
    DATA_GENERATE_NUM = 250
    BASE_LEN_RATIO = 0.3
    SAVE_PATH = "ews_AUC_CI_figs"
    EWS_INDICATORS = ['variance', 'ac1', 'skew', 'kurtosis', 'cv']
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    transition_data, null_data = generate_data(DATA_GENERATE_NUM)
    ews_results = evaluate_ews(transition_data, null_data,
                               ews_indicators=EWS_INDICATORS,
                               save_path=SAVE_PATH,
                               base_len_ratio=BASE_LEN_RATIO)