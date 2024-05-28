# get start
there's two circuit, Elliptic Curve point addition and Fibonacci, use the following code to run the test.

````cargo test --test lib-tests````

# Fibonacci
the struct from 4 and 7 for example:

````4 7````

````7 11````

````11 18````

# Elliptic Curve point addition and Fibonacci
the struct from 4 and 7 for example:
````A = X1 Z22  
   B = X2 Z12 - A  
   c = Y1 Z23
   d = Y2 Z13 - c
   Z3 = Z1 Z2 B
   X3 = d2 - B2 (B + 2 A)
   Y3 = d (A B2  - X3) - c B3````

