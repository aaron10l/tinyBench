from .fi_leakage_topk import compute_answer as compute_fi_leakage_topk
from .fi_leakage_topk_v1 import compute_answer as compute_fi_leakage_topk_v1
from .dq_bad_row_indicator import compute_answer as compute_dq_bad_row_indicator
from .dq_riskier_group import compute_answer as compute_dq_riskier_group
from .rca_retrain_point import compute_answer as compute_rca_retrain_point
from .rca_performance_improve import compute_answer as compute_rca_performance_improve
from .sf_nonmonotone_peak import compute_answer as compute_sf_nonmonotone_peak
from .ix_interaction_dominant import compute_answer as compute_ix_interaction_dominant
from .sf_monotone_classify import compute_answer as compute_sf_monotone_classify

COMPUTE_FN = {
    "fi_leakage_topk_v0": compute_fi_leakage_topk,
    "fi_leakage_topk_v1": compute_fi_leakage_topk_v1,
    "dq_bad_row_indicator_v0": compute_dq_bad_row_indicator,
    "dq_bad_row_indicator_v1": compute_dq_bad_row_indicator,
    "dq_riskier_group_v0": compute_dq_riskier_group,
    "dq_riskier_group_v1": compute_dq_riskier_group,
    "dq_riskier_group_v2": compute_dq_riskier_group,
    "rca_retrain_point_v0": compute_rca_retrain_point,
    "rca_performance_improve_v0": compute_rca_performance_improve,
    "rca_performance_improve_v1": compute_rca_performance_improve,
    "sf_nonmonotone_peak_v0": compute_sf_nonmonotone_peak,
    "ix_interaction_dominant_v0": compute_ix_interaction_dominant,
    "sf_monotone_classify_v0": compute_sf_monotone_classify,
}
