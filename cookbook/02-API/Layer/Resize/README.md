# Resize Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.ResizeMode`
| Name |                             Comment                              |
| :---------------: | :-----------------------------------------------------------: |
|      NEAREST      | Nearest neighbor interpolation, dimensions 1 to 8                |
|      LINEAR       | Linear interpolation, dimensions 1 to 3D |

+ Alternative values of `trt.ResizeCoordinateTransformation`
|  name |          Comment                        |
| :-----------------------------------: | :-----------------------------------------------: |
|             ALIGN_CORNERS             |              |
|              ASYMMETRIC               |  |
|              HALF_PIXEL               |   |

+ About coordinate_transformation

### Notation
+ Uppercase letters denote image grids. For example, $A_{600,800}$ denotes a grid image of height 600 pixels and width 800 pixels.
+ When describing image grids, use square brackets and 2D grid indices starting from zero. Indices can be large and images need not be square. For example, A[1023, 212] denotes the grid point at row 1024 and column 213 of image A.
+ When describing image coordinates, use parentheses and normalized coordinates. Define origin O(0,0) as the top-left corner of grid [0,0] (not the grid center); downward is positive h (1st component), rightward is positive w (2nd component). The bottom-right corner (not center) of the bottom-right grid is Z(1,1). For example, P(1/2,1/3) denotes the point at half height and one-third width from the left.
+ Use v[a,b] to denote the center value of grid [a,b] (i.e., the grid value in source/target image).
+ Use c[a,b], h[a,b], and w[a,b] to denote center coordinate, h-coordinate, and w-coordinate of grid [a,b], with c[a,b] = (h[a,b], w[a,b]).
+ Variables for source image use subscript 1, and variables for target image use subscript 2. For example, $h_{1}, w_{1}, h_{2}, w_{2}$ denote source height/width and target height/width.
+ $i=D_{1}^{n}$ denotes $i=1,2,...,n$, just a shorthand notation.
+ $\lfloor a \rfloor, \lceil a \rceil, \{a\}$ denote floor, ceil, and fractional part respectively ($\{a\}=a-\lfloor a \rfloor$).

### HALF_PIXEL (TensorRT7 align_corners=False)
+ Equivalent to PyTorch (`align_corners=False`).
+ The **four corners (not corner centers)** of source and target images coincide. Build coordinates with the top-left corner as origin, map target grid centers into source coordinates, then perform bilinear interpolation.
    - On source image, $c_{1}\left[i,j\right] = \left(\frac{1}{2h_{1}}+\frac{i}{h_{1}},\frac{1}{2w_{1}}+\frac{j}{w_{1}}\right),i=D_{0}^{h_{1}-1},j=D_{0}^{w_{1}-1}$
    - On target image, $c_{2}\left[i,j\right] = \left(\frac{1}{2h_{1}}+\frac{i}{h_{2}},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\right),i=D_{0}^{h_{2}-1},j=D_{0}^{w_{2}-1}$
    - That is, half a grid height/width is the offset from origin to center of grid (0,0), and adjacent grid centers differ by one grid width.
    - To compute center value at target grid [a,b] (i.e., $v_{2}[a,b]$), find four source grids for interpolation. Let [p,q] be the top-left one, then: $h_{1}[p,q] \le h_{2}[a,b] < h_{1}[p+1,q],\ w_{1}[p,q] \le w_{2}[a,b] < w_{1}[p,q+1]$.
    - Let $\alpha = \frac{h_{1}}{h_{2}}\left(a+\frac{1}{2}\right)-\frac{1}{2},\beta = \frac{w_{1}}{w_{2}}\left(a+\frac{1}{2}\right)-\frac{1}{2}$, then the inequalities are solved by $p = \lfloor \alpha \rfloor, q = \lfloor \beta \rfloor$.
    - Therefore the interpolation result is:
$$
\begin{aligned}
v_{2}[a,b] &=
\left(\frac{h_{1}[p+1,q] - h_{2}[a,b]}{1/h_{1}}\right) \left(\frac{w_{1}[p,q+1] - w_{2}[a,b]}{1/w_{1}}\right) v_{1}[p,q] + \\
&\quad\;\left(\frac{h_{2}[a,b]   - h_{1}[p,q]}{1/h_{1}}\right) \left(\frac{w_{1}[p,q+1] - w_{2}[a,b]}{1/w_{1}}\right) v_{1}[p+1,q] + \\
&\quad\;\left(\frac{h_{1}[p+1,q] - h_{2}[a,b]}{1/h_{1}}\right) \left(\frac{w_{2}[a,b]   - w_{1}[p,q]}{1/w_{1}}\right) v_{1}[p,q+1] + \\
&\quad\;\left(\frac{h_{2}[a,b]   - h_{1}[p,q]}{1/h_{1}}\right) \left(\frac{w_{2}[a,b]   - w_{1}[p,q]}{1/w_{1}}\right) v_{1}[p+1,q+1] \\
&=
\left(1-\{\alpha\}\right)\left(1-\{\beta\}\right) v_{1}[p,q] + \{\alpha\}\left(1-\{\beta\}\right) v_{1}[p+1,q] + \\
&\quad\;\left(1-\{\alpha\}\right)\{\beta\} v_{1}[p,q+1] + \{\alpha\}\{\beta\} v_{1}[p+1,q+1]
\end{aligned}
$$

+ Note: during upsampling, outermost target grid centers may lie outside source outermost grid centers.
    - TensorRT < 7.2: no special handling for outermost centers; formula is applied directly (step1 in figure below).
    - TensorRT >= 7.2: outermost centers are clamped back into the rectangle formed by source outermost centers (step2 in figure below), so spacing at boundary differs from interior spacing.

+ Illustration
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:70%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:70%;" />
</div>
<div align="center" >
<img src="./ResizeLayer-HALF_PIXEL-Step1.png" alt="ResizeLayer-HALF_PIXEL-Step1" style="zoom:70%;" />
<img src="./ResizeLayer-HALF_PIXEL-Step2.png" alt="ResizeLayer-HALF_PIXEL-Step2" style="zoom:70%;" />
</div>

### ALIGN_CORNERS (TensorRT align_corners=True)
+ Equivalent to PyTorch (`align_corners=True`).
+ Align the **centers of the four corner grids** between source and target images. Build coordinates with target top-left corner as origin, map target grid centers into source coordinates, then perform bilinear interpolation.
    - On source image, $c_{1}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{2}}\cdot\frac{h_{2}-1}{h_{1}-1},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\cdot\frac{w_{2}-1}{w_{1}-1}\right),i=D_{0}^{h_{1}-1},j=D_{0}^{w_{1}-1}$
    - On target image, $c_{2}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{2}},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\right),i=D_{0}^{h_{2}-1},j=D_{0}^{w_{2}-1}$
    - Half-grid offset is computed using target image; additionally, because source edge length is scaled, an extra factor is needed (consider cases i=0 and i=h_{1}-1).
    - Computation is similar to above. Let $\alpha = \frac{h_{1}-1}{h_{2}-1}a,\beta = \frac{w_{1}-1}{h_{2}-1}b$, then the inequalities are solved by $p = \lfloor \alpha \rfloor, q = \lfloor \beta \rfloor$.
    - Therefore the interpolation result is still:
$$
\begin{aligned}
v_{2}[a,b] &=
\left(1-\{\alpha\}\right)\left(1-\{\beta\}\right) v_{1}[p,q] + \{\alpha\}\left(1-\{\beta\}\right) v_{1}[p+1,q] + \\
&\quad\;\left(1-\{\alpha\}\right)\{\beta\} v_{1}[p,q+1] + \{\alpha\}\{\beta\} v_{1}[p+1,q+1]
\end{aligned}
$$

+ Illustration
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:70%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:70%;" />
<img src="./ResizeLayer-ALIGN_CORNERS.png" alt="ResizeLayer-ALIGN_CORNERS" style="zoom:70%;" />
</div>

### ASYMMETRIC (TensorRT<7 align_corners=False)
+ Normalize source and target edge lengths to 1, align the **center of top-left grid** between source and target, and do no further scaling. Use target top-left corner as origin, map target grid centers into source coordinates, then perform bilinear interpolation.
    - On source image, $c_{1}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{1}},\frac{1}{2w_{2}}+\frac{j}{w_{1}}\right),i=D_{0}^{h_{1}-1},j=D_{0}^{w_{1}-1}$
    - On target image, $c_{2}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{2}},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\right),i=D_{0}^{h_{2}-1},j=D_{0}^{w_{2}-1}$
    - Half-grid offset uses target image, while adjacent grid spacing is still computed from each image’s own grid counts.
    - Computation is similar to above. Let $\alpha = \frac{h_{1}}{h_{2}}a,\beta = \frac{w_{1}}{h_{2}}b$, then the inequalities are solved by $p = \lfloor \alpha \rfloor, q = \lfloor \beta \rfloor$.
    - Therefore the interpolation result is still:
$$
\begin{aligned}
v_{2}[a,b] &=
\left(1-\{\alpha\}\right)\left(1-\{\beta\}\right) v_{1}[p,q] + \{\alpha\}\left(1-\{\beta\}\right) v_{1}[p+1,q] + \\
&\quad\;\left(1-\{\alpha\}\right)\{\beta\} v_{1}[p,q+1] + \{\alpha\}\{\beta\} v_{1}[p+1,q+1]
\end{aligned}
$$

+ Note: during upsampling, outermost target grid centers may lie outside source outermost grid centers.
    - TensorRT < 7.2: no special handling for outermost centers; formula is applied directly (step1 in figure below).
    - TensorRT >= 7.2: outermost centers are clamped back into the rectangle formed by source outermost centers (step2 in figure below), so spacing at boundary differs from interior spacing.
    -
+ Illustration
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:70%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:70%;" />
</div>
<div align="center" >
<img src="./ResizeLayer-ASYMMETRIC-Step1.png" alt="ResizeLayer-ASYMMETRIC-Step1" style="zoom:70%;" />
<img src="./ResizeLayer-ASYMMETRIC-Step2.png" alt="ResizeLayer-ASYMMETRIC-Step2" style="zoom:70%;" />
</div>
