library(data.table)
library(ggplot2)
library(patchwork)

file_names = list.files("./benchmarks/output", pattern = "solve_ivp_times_(.*).csv",
  full.names = TRUE, include.dirs = TRUE)
bench_dt = rbindlist(lapply(file_names, fread))
bench_dt[, method := factor(method,
  levels = c("PYRICCATICPP", "DOP853", "BDF", "RK45"), ordered=TRUE)]
bremer_dt = bench_dt[eq_name == "BREMER237"]
bremer_dt[, lambda := as.numeric(lapply(strsplit(problem_params, "="), \(x) x[[4]]))]
bremer_dt = bremer_dt[!grep("n=20", params)]
setkey(bremer_dt, method, eps, lambda)
bremer_dt[, relerr := as.numeric(relerr)]
setnafill(bremer_dt, type = "locf", cols = "relerr")
bremer_table_dt = bremer_dt[eps == 1e-12][, .(method, walltime, lambda)]
setkey(bremer_table_dt, lambda, method)
#bremer_table_dt[, relative_time := .SD[, walltime / walltime[3]], .(lambda)]
bremer_table_dt[, method := factor(method,
  levels = c("PYRICCATICPP", "BDF", "DOP853", "RK45"), ordered = TRUE)]
setkey(bremer_table_dt, lambda, method)
bremer_table_format_dt = copy(bremer_table_dt[, .(method, walltime, lambda)])
bremer_table_format_dt = bremer_table_format_dt[,
  .(walltime = median(walltime),
    walltime_sd = sd(walltime)),
  .(method, lambda)]
bremer_table_cast_dt = dcast(bremer_table_format_dt, formula = lambda ~ method, value.var = "walltime")
bremer_table_cast_dt[, `:=`(
  BDF = BDF / PYRICCATICPP,
  DOP853 = DOP853 / PYRICCATICPP,
  RK45 = RK45 / PYRICCATICPP
)]
bremer_table_cast_dt[, PYRICCATICPP := NULL]
bremer_table_format_dt = bremer_table_cast_dt[, .(
  lambda,
  BDF = round(BDF, digits = 2),
  DOP853 = round(DOP853, digits = 2),
  RK45 = round(RK45, digits = 2)
)]
#setcolorder(bremer_table_format_dt, c("method", "lambda"))
knitr::kable(bremer_table_format_dt)
bremer_sum_dt = bremer_dt[,
  .(walltime = median(walltime),
    walltime_sd = sd(walltime),
    solved = sum(success)),.(method, lambda, eps)]
bremer_sum_dt[, all_solved := ifelse(solved == 20,"true", "false"), .I]
bremer_plots = ggplot(bremer_sum_dt,
  aes(x = lambda, y = walltime, color = method)) +
  geom_line() +
  geom_point(aes(shape = all_solved), size = 2.5, show.legend = FALSE) +
  scale_shape_manual(values=c("true"=16, "false"=4),
    labels=c("no","yes"), name="All Runs Successful:") +
  scale_x_log10() +
  scale_y_log10(
    labels = function(x) {
      ifelse(x == 0.0001, format(x, scientific = FALSE), as.character(x))
      },
    breaks = c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000)
  ) +
  facet_wrap(vars(eps)) +
  ggtitle("Bremer eq 237 Median Wall Time in Seconds", "x indicates all runs failed") +
  ylab("") +
  xlab("Bremer Eq 237: Lambda") +
  theme_bw() +
  theme(legend.position="bottom") +
  viridis::scale_color_viridis(discrete = TRUE)
bremer_plots

airy_dt = bench_dt[eq_name == "AIRY"]

stiff_dt = bench_dt[eq_name == "STIFF"]
stiff_dt = stiff_dt[!grep("n=20", params)]
stiff_sum_dt = stiff_dt[,.(
  walltime = median(walltime),
  walltime_sd = sd(walltime),
  solved = sum(success)),.(method, eps)]
stiff_sum_dt[, posit := walltime + walltime * .1]
stiff_sum_dt[method == "BDF" & eps == 1e-6, posit := posit + .01]
stiff_sum_dt[method == "PYRICCATICPP" & eps == 1e-6, posit := posit + .01]
stiff_sum_dt[method == "PYRICCATICPP" & eps == 1e-12, posit := posit + .01]
stiff_sum_dt[method == "RK45" & eps == 1e-6, posit := posit - .01]
stiff_sum_dt[method == "RK45" & eps == 1e-12, posit := posit - .01]
stiff_sum_dt[method == "DOP853" & eps == 1e-6, posit := posit - .02]
stiff_sum_dt[method == "DOP853" & eps == 1e-12, posit := posit - .02]
stiff_plot = ggplot(stiff_sum_dt, aes(x = method, y = walltime,
  fill = method, label = as.character(round(walltime, 4)))) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = posit)) +
  facet_wrap(vars(eps)) +
#  ggtitle("Stiff Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log1p", labels = \(x) as.character(x),
    breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5)) +
  ylab("") +
  xlab("Stiff") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))
stiff_plot

airy_dt = bench_dt[eq_name == "AIRY"]
airy_dt = airy_dt[!grep("n=20", params)]
airy_sum_dt = airy_dt[,.(
  walltime = median(walltime),
  walltime_sd = sd(walltime),
  solved = sum(success)),.(method, eps)]
airy_y_breaks = c(0, 1, 2, 4, 6, 8)
airy_sum_dt[, posit := (walltime) + (walltime) * .1]
airy_sum_dt[method == "BDF" & eps == 1e-12, posit := posit - .2]
airy_sum_dt[method == "PYRICCATICPP" & eps == 1e-6, posit := posit + .1]
airy_sum_dt[method == "PYRICCATICPP" & eps == 1e-12, posit := posit + .1]
airy_sum_dt[method == "DOP853" & eps == 1e-6, posit := posit + .1]
airy_sum_dt[method == "DOP853" & eps == 1e-12, posit := posit + .1]
airy_sum_dt
airy_plot = ggplot(airy_sum_dt, aes(x = method, y = walltime, fill = method)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = as.character(round(walltime , 3)), y = posit)) +
  facet_wrap(vars(eps)) +
#  ggtitle("Airy Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log1p",
    labels = \(x) x, breaks = airy_y_breaks) +
  ylab("") +
  xlab("Airy") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1))
airy_plot

patched_plot = (bremer_plots / (stiff_plot + airy_plot)) + plot_annotation(
  title = 'Benchmarks in Seconds',
  subtitle = 'Benchmarks Faceted by relative tolerance'
)
patched_plot
ggsave("./benchmarks/plots/ivp_bench.png", patched_plot,
  width = 10, height = 8, units = "in")
lseq <- function(from=1, to=100000, length.out=6) {
  # logarithmic spaced sequence
  # blatantly stolen from library("emdbook"), because need only this
  exp(seq(log(from), log(to), length.out = length.out))
}
bremer_sum_dt = bremer_dt[,
  .(relerr = median(relerr),
    relerr_sd = sd(relerr),
    solved = sum(success)),.(method, lambda, eps)]
bremer_sum_dt[, all_solved := ifelse(solved == 20,"true", "false"), .I]

bremer_err_plot = ggplot(bremer_sum_dt,
  aes(x = lambda, y = relerr, color = method, group = method)) +
  facet_wrap(vars(eps), ncol = 1) +
  geom_line() +
  geom_point(aes(shape = all_solved), size = 3, show.legend = FALSE) +
    scale_shape_manual(values=c("true"=16, "false"=4),
      labels=c("no","yes"), name="All Runs Successful:") +
  ylab("") +
  xlab("Bremer: lambda") +
  scale_x_log10() +
  scale_y_log10(breaks = signif(lseq(1e-13, 1, length.out=7),digits = 2)) +
  theme_bw() +
  theme(legend.position="bottom",
    axis.text.y = element_text(size = 12))
bremer_err_plot

cheat_log = scales::new_transform(
  "cheating_log", transform = \(x) ifelse(x > 0, log10(x), x),
  inverse = \(x) ifelse(x > 0, 10^(x), 0)
)
stiff_sum_dt = stiff_dt[,.(
  relerr = median(relerr),
  relerr_sd = sd(relerr),
  solved = sum(success)),.(method, eps)]
stiff_breaks = c(0, 10, 100, 1000, 1e5, 1e7, 1e10)
stiff_err_plot = ggplot(stiff_sum_dt,
  aes(x = method, y = relerr * (1.0e21), fill = method)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(eps)) +
  #  ggtitle("Stiff Equation Wall Time In Seconds") +
  geom_text(aes(label = format(relerr, scientific = TRUE, digits = 3)),
    vjust = -0.4, size = 3) +
  facet_wrap(vars(eps)) +
  scale_y_continuous(transform = cheat_log,
    labels = \(x) x / 1.0e21, breaks = stiff_breaks) +
  ylab("") +
  xlab("Stiff") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1),
    axis.text.y = element_text(size = 12))
stiff_err_plot

airy_sum_dt = airy_dt[,.(
  relerr = median(relerr),
  relerr_sd = sd(relerr),
  solved = sum(success)),.(method, eps)]
airy_err_plot = ggplot(airy_sum_dt, aes(x = method, y = relerr * 1e12, fill = method)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(eps)) +
  geom_text(aes(label = format(relerr, scientific = TRUE, digits = 3)),
    vjust = -0.4, size = 3) +
  #  ggtitle("Stiff Equation Wall Time In Seconds") +
  scale_y_continuous(transform = "log10", labels = \(x) x / 1e12) +
  ylab("") +
  xlab("Airy") +
  theme_bw() +
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = -30, vjust = 0.8, hjust=.1),
    axis.text.y = element_text(size = 12))
airy_err_plot

err_plots = (bremer_err_plot / (stiff_err_plot + airy_err_plot)) + plot_annotation(
  title = 'Relative Error Per Problem',
  subtitle = 'Benchmarks Faceted by relative tolerance'
)
err_plots
ggsave("./benchmarks/plots/ivp_bench_errs.png", err_plots,
  width = 35, height = 25, units = "cm")

