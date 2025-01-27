library(data.table)
library(ggplot2)
library(patchwork)

file_names = list.files("../../../riccaticpp/benchmarks/output", pattern = "solve_ivp_times_(.*).csv",
  full.names = TRUE, include.dirs = TRUE)
bench_dt = rbindlist(lapply(file_names, fread))

bremer_dt = bench_dt[eq_name == "BREMER237"]
bremer_dt[, lambda := as.numeric(lapply(strsplit(problem_params, "="), \(x) x[[4]]))]
bremer_dt = bremer_dt[!grep("n=20", params)]
bremer_plots = ggplot(bremer_dt,
  aes(x = lambda, y = walltime, color = method)) +
  geom_line() +
  geom_point() +
  scale_x_log10() +
  scale_y_log10(
    labels = function(x) {
      ifelse(x == 0.0001, format(x, scientific = FALSE), as.character(x))
      },
    breaks = c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000)
  ) +
  facet_wrap(vars(eps)) +
#  ggtitle("Bremer eq 237 Wall Time in Seconds") +
  ylab("") +
  xlab("Bremer Eq 237: Lambda") +
  theme_bw() +
  theme(legend.position="bottom")

bremer_plots

airy_dt = bench_dt[eq_name == "AIRY"]

stiff_dt = bench_dt[eq_name == "STIFF"]
stiff_dt = stiff_dt[!grep("n=20", params)]

stiff_plot = ggplot(stiff_dt, aes(x = method, y = walltime, fill = method)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(eps)) +
#  ggtitle("Stiff Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log1p", breaks = c(0, 0.05, 0.1, 0.3, 0.6)) +
  ylab("") +
  xlab("Stiff") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))
stiff_plot
airy_dt = bench_dt[eq_name == "AIRY"]
airy_dt = airy_dt[!grep("n=20", params)]

airy_plot = ggplot(airy_dt, aes(x = method, y = walltime, fill = method)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(eps)) +
#  ggtitle("Airy Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log1p", breaks = c(0, 0.1, 0.5, 3, 6)) +
  ylab("") +
  xlab("Airy") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))
airy_plot

(bremer_plots / (stiff_plot + airy_plot)) + plot_annotation(
  title = 'Benchmarks in Seconds',
  subtitle = 'Benchmarks Faceted by relative tolerance'
)

setkey(bremer_dt, method, eps, lambda)
bremer_dt[, relerr := nafill(relerr, type = "locf"), .(method, eps)]
bremer_err_plot = ggplot(bremer_dt, aes(x = lambda, y = relerr, color = method, group = method)) +
  facet_wrap(vars(eps), ncol = 1) +
  geom_point() +
  geom_line() +
  ylab("") +
  xlab("Bremer: lambda") +
  scale_x_log10() +
  scale_y_log10(breaks = c(1e-12, 1e-10, 1e-06, 1e-2, 1)) +
  theme_bw() +
  theme(legend.position="bottom",
    axis.text.y = element_text(size = 12))
bremer_err_plot

stiff_err_plot = ggplot(stiff_dt, aes(x = method, y = relerr, fill = method)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(eps)) +
  #  ggtitle("Stiff Equation Wall Time In Seconds") +
  geom_text(aes(label = format(relerr, scientific = TRUE, digits = 3)), vjust = -0.4) +
  facet_wrap(vars(eps), scales = "free_y") +
  scale_y_continuous(transform = "log1p", labels = scales::scientific_format()) +
  ylab("") +
  xlab("Stiff") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1),
    axis.text.y = element_text(size = 12))


airy_err_plot = ggplot(airy_dt, aes(x = method, y = relerr, fill = method)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = format(relerr, scientific = TRUE, digits = 3)), vjust = -0.4) +
  facet_wrap(vars(eps), scales = "free_y") +
  #  ggtitle("Stiff Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log1p",labels = scales::scientific_format()) +
  ylab("") +
  xlab("Airy") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1),
    axis.text.y = element_text(size = 12))

(bremer_err_plot / (stiff_err_plot + airy_err_plot)) + plot_annotation(
  title = 'Relative Error Per Problem',
  subtitle = 'Benchmarks Faceted by relative tolerance'
)

