# get start
there's two circuit, Elliptic Curve point addition and Fibonacci, use the following code to run the test.

````cargo test --test lib-tests````

# Fibonacci
the struct from 4 and 7 for example:

````4 7````

````7 11````

````11 18````

# Elliptic Curve point addition and Fibonacci
jacobian coordinates addition:
````
   point1 = (x1,y1,z1)
   point2 = (x2,y2,z2)
   point1 + point2 = point3
   A = X1 * Z2²  
   B = X2 * Z1² - A  
   c = Y1 * Z2³
   d = Y2 * Z1³ - c
   Z3 = Z1 * Z2 * B
   X3 = d2 - B²*(B + 2 A)
   Y3 = d (A*B²  - X3) - c*B³  

