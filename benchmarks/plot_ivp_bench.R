library(data.table)
library(ggplot2)
library(patchwork)

file_names = list.files("../../../riccaticpp/benchmarks/output/", pattern = "*.csv",
  full.names = TRUE, include.dirs = TRUE)
bench_dt = rbindlist(lapply(file_names[1:4], fread))

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
    breaks = c(0.0001, 0.001, 0.1, 10, 1000)
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
  facet_wrap(vars(eps), scales = "free_y") +
#  ggtitle("Airy Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log1p") +
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
