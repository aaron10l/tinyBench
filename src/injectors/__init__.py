from .fi_leakage_topk import inject as inject_fi_leakage_topk
from .dq_riskier_group import inject as inject_dq_riskier_group
from .dq_data_quality_filter import inject as inject_dq_data_quality_filter
from .dq_data_quality_filter_v1 import inject as inject_dq_data_quality_filter_v1
from .rca_performance_improve import inject as inject_rca_performance_improve
from .rca_retrain_point import inject as inject_rca_retrain_point
from .sf_nonmonotone_peak import inject as inject_sf_nonmonotone_peak
from .ix_interaction_dominant import inject as inject_ix_interaction_dominant
from .sf_monotone_classify import inject as inject_sf_monotone_classify

INJECT_FN = {
    "fi_leakage_topk": inject_fi_leakage_topk,
    "dq_riskier_group": inject_dq_riskier_group,
    "dq_data_quality_filter": inject_dq_data_quality_filter,
    "dq_data_quality_filter_v1": inject_dq_data_quality_filter_v1,
    "rca_performance_improve": inject_rca_performance_improve,
    "rca_retrain_point": inject_rca_retrain_point,
    "sf_nonmonotone_peak": inject_sf_nonmonotone_peak,
    "ix_interaction_dominant": inject_ix_interaction_dominant,
    "sf_monotone_classify": inject_sf_monotone_classify,
}
