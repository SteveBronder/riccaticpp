installed_packages = installed.packages()
installed_packages = installed_packages[,"Package"]
needed_packages = c("data.table", "ggplot2", "patchwork", "knitr", "viridis", "scales")

to_be_installed = needed_packages[!(needed_packages %in% installed_packages)]
if (length(to_be_installed) > 0) {
  install.packages(to_be_installed)
}
