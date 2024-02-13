#!/usr/bin/env python3

""" Скрипт запуска.
"""

from runtime.api.adapter import Adapter
from runtime.lib.utilities.env_utility import EnvUtility

if __name__ == "__main__":
  adapter: Adapter = Adapter(
    enable_swagger=EnvUtility().get_bool(var_name="CORTEX_API_ENABLE_SWAGGER")
  )

  adapter.run_server(
    host=EnvUtility().get_str(var_name="CORTEX_API_HOST", default_value="0.0.0.0"),
    port=EnvUtility().get_int(var_name="CORTEX_API_PORT", default_value=8000),
    routers_file="routers.yml"
  )
