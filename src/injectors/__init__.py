from .name_swap_injection import inject as inject_name_swap
from .unit_scale_injection import inject as inject_unit_scale
from .heteroskedastic_injection import inject as inject_heteroskedastic
from .bad_rows_injection import inject as inject_bad_rows
from .simpsons_paradox_injection import inject as inject_simpsons_paradox
from .changepoint_injection import inject as inject_changepoint

INJECT_FN = {
    "name_swap_injection": inject_name_swap,
    "unit_scale_injection": inject_unit_scale,
    "heteroskedastic_injection": inject_heteroskedastic,
    "bad_rows_injection": inject_bad_rows,
    "simpsons_paradox_injection": inject_simpsons_paradox,
    "changepoint_injection": inject_changepoint,
}
