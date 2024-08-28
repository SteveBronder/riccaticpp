## Airy Example

An an example we will solve the airy equation where the omega and gamma functions are given by the following.

$$
\omega(x) = \sqrt(x)
\gamma(x) = 0
$$

From the top level folder we can call the following to run cmake, make, and run the example.

```bash
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE  -DRICCATI_BUILD_EXAMPLES=ON 
cd build/examples
make airy_eq && ./airy_eq
```