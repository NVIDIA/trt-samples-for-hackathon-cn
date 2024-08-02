# Loop Structure

+ Steps to run.

```bash
python3 main.py
```

+ The Loop structure contains usage of `LoopOutputLayer` and `TripLimitLayer`.

+ Alternative values of tensorrt.TripLimit
| name  |  Comment   |
| :---: | :--------: |
| COUNT |  for loop  |
| WHILE | while loop |

+ Alternative values of tensorrt.LoopOutput
|    name     |                Comment                |
| :---------: | :-----------------------------------: |
| LAST_VALUE  |         Keep the last output          |
| CONCATENATE | Keep all output in forward direction |
|   REVERSE   | Keep all output in reverse direction |

+ Unidirectional LSTM

  + Output 0: y
    $$
    \left[\begin{matrix}
        0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 \\
        0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 \\
        0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
    \end{matrix}\right]
    $$

  + Output 1: h1
  $$
  \left[\begin{matrix}
      \left[\begin{matrix}
          {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} \\
          {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} \\
          0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 \\
          0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
      \end{matrix}\right] \\
      \left[\begin{matrix}
          0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 \\
          0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 \\
          0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 \\
          0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
      \end{matrix}\right] \\
      \left[\begin{matrix}
          0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 \\
          0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 \\
          0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 \\
          0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
      \end{matrix}\right]
  \end{matrix}\right]
  $$

  + Output 2: c1
  $$
  \left[\begin{matrix}
      3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906 \\
      3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906 \\
      3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906
  \end{matrix}\right]
  $$

+ Process of computation (Here $b_{*} = b_{*,X} + b_{?,H}$, "$*$" can be I, F or O)

$$
\begin{aligned}
I_{1} = F_{1} = O_{1} = \textbf{sigmoid} \left( W_{*,X} \cdot x_{1} + W_{*,H} \cdot h_{0} + b_{*} \right) &=
    \left( 0.99999386,0.99999386,0.99999386,0.99999386,0.99999386 \right) ^\mathrm{T} \\
C_{1}=\textbf{tanh} \left( W_{C,X}\cdot x_{1}+W_{C,H}\cdot h_{0}+b_{C} \right) &=
    \left( 0.99999999,0.99999999,0.99999999,0.99999999,0.99999999 \right) ^\mathrm{T} \\
c_{1} = F_{1} \cdot c_{0} + I_{1} \cdot C_{1} &=
    \left( 0.99999386,0.99999386,0.99999386,0.99999386,0.99999386 \right) ^\mathrm{T} \\
h_{1} = O_{1} \cdot \textbf{tanh} \left( c_{1} \right) &=
    \left(
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}}
    \right) ^\mathrm{T} \\
\hfill \\
I_{2} = F_{2} = O_{2} = \textbf{sigmoid} \left( W_{*,X} \cdot x_{2} + W_{*,H} \cdot h_{1} + b_{*} \right) &=
    \left( 0.99997976,0.99997976,0.99997976,0.99997976,0.99997976 \right) ^\mathrm{T} \\
C_{2} = \textbf{tanh} \left( W_{C,X} \cdot x_{2} + W_{C,H} \cdot h_{1} + b_{C} \right) &=
    \left( 0.99999999,0.99999999,0.99999999,0.99999999,0.99999999 \right) ^\mathrm{T} \\
c_{2} = F_{2} \cdot c_{1} + I_{2} \cdot C_{2} &=
    \left( 1.99995338,1.99995338,1.99995338,1.99995338,1.99995338 \right) ^\mathrm{T} \\
h_{2} = O_{2} \cdot \textbf{tanh} \left( c_{2} \right) &=
    \left(
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}}
    \right) ^\mathrm{T} \\
\end{aligned}
$$
