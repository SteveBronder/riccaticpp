# pyriccaticpp benchmarks

## Running Benchmarks

From the top level run the following to create the timing and performance output files
```python
pip install -r ./requirements.txt
python3 ./benchmarks/schrodinger_eq.py
python3 ./benchmarks/solve_ivp_bench.py
```

## Solve IVP Results

Times for each graph are on a logarithmic scale. For Bremer, even in the large lambda case the 2nd fastest solver relative to pyriccaticpp is still 4.7x

![ivp_bench](/benchmarks/plots/ivp_bench.png)

### Table For Bremer Eq 237 BDF Relative Wall Time To Pyriccaticpp

|Method       | Lambda| Wall Time   | Relative Time|
|:------------|------:|:------------|-------------:|
|BDF          |  1e+01|2.517111e-01 |         70.76|
|BDF          |  1e+02|2.217773e+00 |      17436.49|
|BDF          |  1e+03|6.117332e-02 |        567.63|
|BDF          |  1e+04|6.243414e-02 |        605.13|
|BDF          |  1e+05|1.286601e-03 |         12.38|
|BDF          |  1e+06|7.355530e-04 |          7.21|
|BDF          |  1e+07|5.904986e-04 |          6.01|

![ivp_bench_err](/benchmarks/plots/ivp_bench_errs.png)

asdfasdf

## Schrödinger Equation

![sch_bench](/benchmarks/plots/schrodinger.png)

![sch_bench_err](/benchmarks/plots/schrodinger_err.png)
