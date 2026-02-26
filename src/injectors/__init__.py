from .fi_leakage_topk import inject as inject_fi_leakage_topk
from .anomaly_riskier_group import inject as inject_anomaly_riskier_group
from .anomaly_data_quality_filter import inject as inject_anomaly_data_quality_filter
from .anomaly_data_quality_filter_v1 import inject as inject_anomaly_data_quality_filter_v1
from .rca_performance_improve import inject as inject_rca_performance_improve
from .rca_retrain_point import inject as inject_rca_retrain_point

INJECT_FN = {
    "fi_leakage_topk": inject_fi_leakage_topk,
    "anomaly_riskier_group": inject_anomaly_riskier_group,
    "anomaly_data_quality_filter": inject_anomaly_data_quality_filter,
    "anomaly_data_quality_filter_v1": inject_anomaly_data_quality_filter_v1,
    "rca_performance_improve": inject_rca_performance_improve,
    "rca_retrain_point": inject_rca_retrain_point,
}
