# Unary Layer

+ Steps to run.

```bash
python3 main.py
```

+ Use Activation Layer for $tanh$.

+ Alternative values of `trt.UnaryOperation`

| Name  |                           Comment                            |
| :---: | :----------------------------------------------------------: |
|  ABS  |                      $\lvert x \rvert$                       |
| ACOS  |                 $\cos^{-1} \left( x \right)$                 |
| ACOSH |                $\cosh^{-1} \left( x \right)$                 |
| ASIN  |                 $\sin^{-1} \left( x \right)$                 |
| ASINH |                $\sinh^{-1} \left( x \right)$                 |
| ATAN  |                 $\tan^{-1} \left( x \right)$                 |
| ATANH |                $\tanh^{-1} \left( x \right)$                 |
| CEIL  |                   $x - \lfloor x \rfloor$                    |
|  COS  |                   $\cos \left( x \right)$                    |
| COSH  |                   $\cosh \left( x \right)$                   |
|  ERF  | $erf \left( x \right) = \int_{0}^{x} \exp\left(-t^{2}\right)dt$ |
|  EXP  |                   $\exp \left( x \right)$                    |
| FLOOR |                     $\lfloor x \rfloor$                      |
| ISINF |                  1 if element == Inf else 0                  |
| ISNAN |                  1 if element == NaN else 0                  |
|  LOG  |                   $\log \left( x \right)$                    |
|  NEG  |                             $-x$                             |
|  NOT  |            $not \left( x \right)$ (only for BOOL)            |
| RECIP |                        $\frac{1}{x}$                         |
| ROUND |                    Round$\left(x\right)$                     |
| SIGN  |             $\frac{1}{1+\exp{\left(-x\right)}}$              |
|  SIN  |                   $\sin \left( x \right)$                    |
| SINH  |                   $\sinh \left( x \right)$                   |
| SQRT  |                          $\sqrt{x}$                          |
|  TAN  |                   $\tan \left( x \right)$                    |
