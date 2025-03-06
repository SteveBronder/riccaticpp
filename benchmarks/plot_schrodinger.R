library(data.table)
library(ggplot2)
library(patchwork)

bench_dt = fread("./benchmarks/output/schrodinger_times2.csv")
bench_dt = unique(bench_dt)
bench_dt[, algo := strsplit(name, " ")[[1]][1], name]
algo_args = \(x) {
  return(paste0("[", strsplit(x, "\\[",perl = TRUE)[[1]][2]))
}
bench_dt[, algo_args := algo_args(name), .I]

prob_args = \(x) {
  return(paste0("[", strsplit(x, "\\]\\[", perl = TRUE)[[1]][2]))
}
bench_dt[1, prob_args(name)]
bench_dt[, prob_args := prob_args(name), .I]
bench_dt[, name := NULL]
bench_dt[grepl("1e-06", algo_args), eps := 1e-06]
bench_dt[!grepl("1e-06", algo_args), eps := 1e-12]
bench_dt[, algo_plus := algo]
bench_dt[grepl("n=20", algo_args), algo_plus := paste0(algo, ":n=20")]
bench_dt[grepl("n=35", algo_args), algo_plus := paste0(algo, ":n=35")]
bench_dt = bench_dt[!grepl("n=20", algo_args)]
setkey(bench_dt, eps, algo_plus)
bench_sum_dt = bench_dt[, .(
  sum_time = sum(time),
  sum_count = sum(count),
  sd_time = sd(time),
  sd_count = sd(count)),
  .(algo, algo_plus, algo_args, prob_args, eps)]
bench_sum_dt[, per_call := sum_time/sum_count]
setkey(bench_sum_dt, eps, algo_plus)
schrod_bench_plot = ggplot(bench_sum_dt, aes(x = algo, y = per_call, fill = algo)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(transform = "log1p", breaks = c(0.1, 2, 8, 30, 128, 500, 2000)) +
  facet_wrap(vars(eps)) +
  ggtitle("Schrodinger: Average Seconds For ODE Solver Calls",
    "Average Time is over all optimizations for each quantum number") +
  xlab("") +
  ylab("") +
  theme_bw() +
  theme(legend.position="bottom") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))
schrod_bench_plot
ggsave("./benchmarks/plots/schrodinger.png", schrod_bench_plot,
  width = 6, height = 4, units = "in")

bench_sum_dt = bench_dt[, .(
  sum_time = sum(time),
  sum_count = sum(count),
  mean_time = mean(time),
  mean_count = mean(count),
  mean_per_time = mean(time / count),
  sd_per_time = sd(time / count),
  sd_time = sd(time),
  sd_count = sd(count)),
  .(algo, algo_plus, algo_args, eps, prob_args)]
bench_sum_dt[, per_call := sum_time / sum_count]
bench_sum_dt[, lb := as.numeric(sub(".*\\blb=([0-9]+).*", "\\1", prob_args)), .I]
bench_sum_dt[, rb := as.numeric(sub(".*\\brb=([0-9]+).*", "\\1", prob_args)), .I]
bench_sum_dt[, prob_bounds := paste0(lb, "-", rb)]
bench_sum_dt[, prob_bounds := factor(prob_bounds, ordered = TRUE, levels =
    c("416-417", "1035-1037", "21930-21940", "471100-471110"))]
bench_sum_dt[, quantum_number := rep(c(50, 100, 1000, 10000),8)]
bench_sum_table_dt = bench_sum_dt[eps == 1e-6, .(algo, quantum_number, per_call)]
bench_sum_table_dt = dcast(bench_sum_table_dt, quantum_number ~ algo, value.var = "per_call")
bench_sum_table_dt[, `:=`(
  BDF = BDF / PYRICCATICPP,
  DOP853 = DOP853 / PYRICCATICPP,
  RK45 = RK45 / PYRICCATICPP
)]
bench_sum_table_dt[, PYRICCATICPP := NULL]
bench_sum_table_dt = bench_sum_table_dt[, lapply(.SD, \(x) format(x, digits = 4))]
knitr::kable(bench_sum_table_dt)
bench_sum_x_scale_dt = bench_sum_dt[1:4]
bench_sum_x_scale_dt[, quantum_number := c(50, 100, 1000, 10000)]
bench_per_energy_plot = ggplot(bench_sum_dt, aes(x = lb, y = mean_per_time, group = algo)) +
  geom_ribbon(aes(ymin = mean_per_time - 2 * sd_per_time, ymax = mean_per_time + 2 * sd_per_time), fill = "grey70") +
  geom_line(aes(color = algo)) +
  geom_point(aes(color = algo)) +
  ggtitle("Schrodinger: Average Seconds For ODE Solver Calls By",
    "By Quantum Number") +
  scale_y_log10(breaks = c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 50)) +
  scale_x_log10(breaks = bench_sum_x_scale_dt[, lb], labels = bench_sum_x_scale_dt[, quantum_number]) +
  xlab("") +
  ylab("") +
  facet_wrap(vars(eps)) +
  theme_bw() +
  theme(legend.position="bottom")
bench_per_energy_plot
ggsave("./benchmarks/plots/schrodinger_energy.png", bench_per_energy_plot,
  width = 6, height = 4, units = "in")

bench_sum_dt[eps == 1e-12 & grepl("lb=471100;rb=471110", prob_args)]
# Error rates
err_dt = fread("./benchmarks/output/schrod2.csv")
err_dt[, algo := strsplit(name, " ")[[1]][1], name]
err_dt[, algo_args := algo_args(name), .I]
err_dt[, prob_args := prob_args(name), .I]
err_dt[grepl("1e-06", algo_args), eps := 1e-06]
err_dt[!grepl("1e-06", algo_args), eps := 1e-12]
err_dt[, algo_plus := algo]
err_dt[grepl("n=20", algo_args), algo_plus := paste0(algo, ":n=20")]
err_dt[grepl("n=35", algo_args), algo_plus := paste0(algo, ":n=35")]
err_dt[, rel_energy_err := abs(energy_error / energy_reference)]
# Just look at a slice
err_sub_dt = err_dt[energy_reference == 471103.777]
#err_sub_dt[, energy_reference := 471103.777]
setkey(err_sub_dt, eps, algo_plus)
knitr::kable(err_sub_dt[, .(algo_plus, algo_args, energy, energy_reference, energy_error, rel_energy_err)])

setkey(err_dt, name)
err_dt[, iter := NULL]
err_dt = unique(err_dt)
err_summary_dt = err_dt[,
  .(total_err = mean(energy_error),
    err_val = mean(rel_energy_err)),
  .(algo, algo_args, prob_args, eps)]
err_summary_dt = err_summary_dt[!grepl("n=20", algo_args)]
setkey(err_summary_dt, algo, eps, prob_args)
err_summary_dt[, lb := as.numeric(sub(".*\\blb=([0-9]+).*", "\\1", prob_args)), .I]
err_summary_dt[, rb := as.numeric(sub(".*\\brb=([0-9]+).*", "\\1", prob_args)), .I]
err_summary_dt[, prob_bounds := paste0(lb, "-", rb)]
err_summary_dt[, prob_bounds := factor(prob_bounds, ordered = TRUE, levels =
    c("416-417", "1035-1037", "21930-21940", "471100-471110"))]

err_summary_dt[, md := ((lb + rb) / 2.0) + lb]
err_summary_dt[, blah := as.numeric(as.factor(algo))]
setkey(err_summary_dt, eps, algo, algo_args, prob_bounds)
err_summary_dt[, bound_idx := as.numeric(prob_bounds)]
err_summary_dt[, test := seq(from = bound_idx - .1, to = bound_idx + .1, length.out = 4)[blah], .I]
err_summary_dt[rb == 417]
setkey(err_summary_dt, eps, algo, algo_args, prob_bounds)
err_summary_dt
err_summary_dt[, min(err_val)]
# We multiply by 1_000_000 and then divide so we can make the y axis logarithmic
# While still looking nice
schrod_err_plot = ggplot(err_summary_dt[algo != "BDF"],
  aes(x = test, y = err_val, color = algo, fill = algo)) +
  #  scale_y_log10(breaks = c(1, 2, 5, 10, 18, 30, 60)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(eps), nrow = 2, ncol = 1) +
  ggtitle("Schrodinger: Relative Error of Energy Per ODE",
    "BDF excluded due to too large of a relative error") +
  scale_x_continuous(labels = c(50, 100, 1000, 10000)) +
  scale_y_continuous(breaks = c(1e-8, 1.5e-7, 4e-7, 6e-7)) +
  xlab("Quantum Number") +
  ylab("") +
  theme_bw() +
  theme(legend.position="bottom", axis.text.y = element_text(size = 12))
schrod_err_plot
ggsave("./benchmarks/plots/schrodinger_err.png", schrod_err_plot,
  width = 6, height = 4, units = "in")

