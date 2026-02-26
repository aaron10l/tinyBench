from .fi_leakage_topk import compute_answer as compute_fi_leakage_topk
from .anomaly_data_quality_filter import compute_answer as compute_anomaly_data_quality_filter
from .anomaly_data_quality_filter_v1 import compute_answer as compute_anomaly_data_quality_filter_v1
from .anomaly_riskier_group import compute_answer as compute_anomaly_riskier_group
from .rca_retrain_point import compute_answer as compute_rca_retrain_point
from .rca_performance_improve import compute_answer as compute_rca_performance_improve

COMPUTE_FN = {
    "fi_leakage_topk_v0": compute_fi_leakage_topk,
    "anomaly_data_quality_filter_v0": compute_anomaly_data_quality_filter,
    "anomaly_data_quality_filter_v1": compute_anomaly_data_quality_filter_v1,
    "anomaly_riskier_group_v0": compute_anomaly_riskier_group,
    "rca_retrain_point_v0": compute_rca_retrain_point,
    "rca_performance_improve_v0": compute_rca_performance_improve,
}
