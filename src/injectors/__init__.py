from .fi_top_features import inject as inject_fi_top_features
from .dq_riskier_group import inject as inject_dq_riskier_group
from .dq_bad_row_indicator import inject as inject_dq_bad_row_indicator
from .dq_bad_row_indicator_v1 import inject as inject_dq_bad_row_indicator_v1
from .rca_performance_improve import inject as inject_rca_performance_improve
from .rca_retrain_point import inject as inject_rca_retrain_point
from .sf_nonmonotone_peak import inject as inject_sf_nonmonotone_peak
from .ix_interaction_dominant import inject as inject_ix_interaction_dominant
from .sf_monotone_classify import inject as inject_sf_monotone_classify

INJECT_FN = {
    "fi_top_features": inject_fi_top_features,
    "dq_riskier_group": inject_dq_riskier_group,
    "dq_bad_row_indicator": inject_dq_bad_row_indicator,
    "dq_bad_row_indicator_v1": inject_dq_bad_row_indicator_v1,
    "rca_performance_improve": inject_rca_performance_improve,
    "rca_retrain_point": inject_rca_retrain_point,
    "sf_nonmonotone_peak": inject_sf_nonmonotone_peak,
    "ix_interaction_dominant": inject_ix_interaction_dominant,
    "sf_monotone_classify": inject_sf_monotone_classify,
}
