library(data.table)
library(ggplot2)
library(patchwork)

bench_dt = fread("../../../riccaticpp/benchmarks/output/schrodinger_times.csv")
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
bench_dt[grep("1e-06", algo_args), eps := 1e-06]
bench_dt[!grep("1e-06", algo_args), eps := 1e-12]
bench_dt[, algo_plus := algo]
bench_dt[grep("n=20", algo_args), algo_plus := paste0(algo, ":n=20")]
bench_dt[grep("n=35", algo_args), algo_plus := paste0(algo, ":n=35")]
bench_dt = bench_dt[!grep("n=20", algo_args)]
setkey(bench_dt, eps, algo_plus)
bench_sum_dt = bench_dt[, .(sum_time = sum(time), sum_count = sum(count)),
  .(algo, algo_plus, algo_args, eps)]
bench_sum_dt[, per_call := sum_time/sum_count]
setkey(bench_sum_dt, eps, algo_plus)
ggplot(bench_sum_dt, aes(x = algo_plus, y = per_call, fill = algo)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(transform = "log1p", breaks = c(0, 0.25, .5, 1, 2, 4, 8, 13)) +
  facet_wrap(vars(eps)) +
  ggtitle("Schrodinger Equation: Average Seconds For ODE Solves") +
  xlab("") +
  ylab("") +
  theme_bw() +
  theme(legend.position="bottom") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))


# Error rates
err_dt = fread("../../../riccaticpp/benchmarks/output/schrod.csv")
err_dt[, algo := strsplit(name, " ")[[1]][1], name]
err_dt[, algo_args := algo_args(name), .I]
err_dt[, prob_args := prob_args(name), .I]
err_dt[grep("1e-06", algo_args), eps := 1e-06]
err_dt[!grep("1e-06", algo_args), eps := 1e-12]
err_dt[, algo_plus := algo]
err_dt[grep("n=20", algo_args), algo_plus := paste0(algo, ":n=20")]
err_dt[grep("n=35", algo_args), algo_plus := paste0(algo, ":n=35")]
setkey(err_dt, name)
err_summary_dt = err_dt[,
  .(err_avg = mean(energy_error)),
  .(algo, algo_args, prob_args, eps)]

setkey(err_summary_dt, algo, eps, prob_args)
err_summary_dt[, lb := sub(".*\\blb=([0-9]+).*", "\\1", prob_args), .I]
err_summary_dt[, rb := sub(".*\\brb=([0-9]+).*", "\\1", prob_args), .I]
err_summary_dt[, prob_bounds := paste0(lb, "-", rb)]
ggplot(err_summary_dt, aes(x = prob_bounds, y = err_avg, color = algo, group = algo)) +
  geom_line() +
  geom_point() +
  #  scale_y_log10(breaks = c(1, 2, 5, 10, 18, 30, 60)) +
  facet_wrap(vars(eps), nrow = 2, ncol = 1) +
  ggtitle("Schrodinger Equation: Energy Error Per ODE") +
  xlab("") +
  ylab("") +
  theme_bw() +
  theme(legend.position="bottom")
#  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))
