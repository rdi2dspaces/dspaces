set(DSPACES_INSTALL_MODULE_PATH "${CMAKE_INSTALL_PREFIX}/share/modules")
set(script-mods s3nc_mod.py azure_mod.py url_mod.py ds_reg.py)

install(FILES ${script-mods} DESTINATION ${DSPACES_INSTALL_MODULE_PATH})
