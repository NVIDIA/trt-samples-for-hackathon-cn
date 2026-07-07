# Unary layer

+ Apply an element-wise unary operation to the input tensor.

+ Steps to run.

```bash
python3 main.py
```

+ Use the Activation layer for $\tanh$ (there is no TANH unary operation).

+ Available values of `trt.UnaryOperation`.

| Name  |                                          Comment                                          |
| :---: | :---------------------------------------------------------------------------------------: |
|  EXP  |                                  $\exp \left( x \right)$                                  |
|  LOG  |                                  $\log \left( x \right)$                                  |
| SQRT  |                                        $\sqrt{x}$                                         |
| RECIP |                                       $\frac{1}{x}$                                       |
|  ABS  |                                     $\lvert x \rvert$                                     |
|  NEG  |                                           $-x$                                            |
|  SIN  |                                  $\sin \left( x \right)$                                  |
|  COS  |                                  $\cos \left( x \right)$                                  |
|  TAN  |                                  $\tan \left( x \right)$                                  |
| SINH  |                                 $\sinh \left( x \right)$                                  |
| COSH  |                                 $\cosh \left( x \right)$                                  |
| ASIN  |                               $\sin^{-1} \left( x \right)$                                |
| ACOS  |                               $\cos^{-1} \left( x \right)$                                |
| ATAN  |                               $\tan^{-1} \left( x \right)$                                |
| ASINH |                               $\sinh^{-1} \left( x \right)$                               |
| ACOSH |                               $\cosh^{-1} \left( x \right)$                               |
| ATANH |                               $\tanh^{-1} \left( x \right)$                               |
| CEIL  |                                     $\lceil x \rceil$                                     |
| FLOOR |                                    $\lfloor x \rfloor$                                    |
|  ERF  | $\mathrm{erf}\left(x\right) = \frac{2}{\sqrt{\pi}}\int_{0}^{x} \exp\left(-t^{2}\right)dt$ |
|  NOT  |                              $\lnot x$ (only for BOOL input)                              |
| SIGN  |         $\mathrm{sign}\left(x\right)$ (-1, 0, or 1 depending on the sign of $x$)          |
| ROUND |              $\mathrm{round}\left(x\right)$ (round to nearest, ties to even)              |
| ISINF |                          1 if the element is $\pm\infty$ else 0                           |
| ISNAN |                              1 if the element is NaN else 0                               |
