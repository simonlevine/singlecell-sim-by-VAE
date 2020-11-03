from pipeline.helpers.paths import PARAMS_FP
from dynaconf import Dynaconf

params = Dynaconf(
    envvar_prefix="VAE_",
    settings_files=[PARAMS_FP],
)