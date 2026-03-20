from .fi_leakage_topk import compute_answer as compute_fi_leakage_topk
from .fi_leakage_topk_v1 import compute_answer as compute_fi_leakage_topk_v1
from .anomaly_data_quality_filter import compute_answer as compute_anomaly_data_quality_filter
from .anomaly_riskier_group import compute_answer as compute_anomaly_riskier_group
from .rca_retrain_point import compute_answer as compute_rca_retrain_point
from .rca_performance_improve import compute_answer as compute_rca_performance_improve
from .fi_nonmonotone_peak import compute_answer as compute_fi_nonmonotone_peak
from .fi_interaction_dominant import compute_answer as compute_fi_interaction_dominant

COMPUTE_FN = {
    "fi_leakage_topk_v0": compute_fi_leakage_topk,
    "fi_leakage_topk_v1": compute_fi_leakage_topk_v1,
    "anomaly_data_quality_filter_v0": compute_anomaly_data_quality_filter,
    "anomaly_data_quality_filter_v1": compute_anomaly_data_quality_filter,
    "anomaly_riskier_group_v0": compute_anomaly_riskier_group,
    "anomaly_riskier_group_v1": compute_anomaly_riskier_group,
    "anomaly_riskier_group_v2": compute_anomaly_riskier_group,
    "rca_retrain_point_v0": compute_rca_retrain_point,
    "rca_performance_improve_v0": compute_rca_performance_improve,
    "rca_performance_improve_v1": compute_rca_performance_improve,
    "fi_nonmonotone_peak_v0": compute_fi_nonmonotone_peak,
    "fi_interaction_dominant_v0": compute_fi_interaction_dominant,
}
