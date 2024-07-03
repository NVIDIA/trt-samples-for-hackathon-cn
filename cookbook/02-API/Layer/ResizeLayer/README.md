# Resize Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of trt.ResizeMode
| Name |                             Comment                              |
| :---------------: | :-----------------------------------------------------------: |
|      NEAREST      | Nearest neighbor interpolation, dimensions 1 to 8                |
|      LINEAR       | Linear interpolation, dimensions 1 to 3D |

+ Alternative values of trt.ResizeCoordinateTransformation
|  name |          Comment                        |
| :-----------------------------------: | :-----------------------------------------------: |
|             ALIGN_CORNERS             |              |
|              ASYMMETRIC               |  |
|              HALF_PIXEL               |   |

+ 关于 coordinate_transformation

### 符号约定
+ 大写字母表示图像栅格，入如 $A_{600,800}$ 表示一张高 600 像素，宽 800 像素的栅格图
+ 描述图像栅格时使用中括号、格点二维索引，从零开始计数，索引可以很大且图像可以不是正方形。如 A[1023, 212] 表示图 A（首行为第 1 行）的第 1024 行和（首列为第 1 列）的第 213 列的格点
+ 描述图像坐标时使用小括号、归一化坐标，规定原点 O(0,0) 为 [0,0] 栅格的左上角角落（不是栅格中心），数值向下为 h 正方向 （第 1 分量），水平向右为 w 正方向 （第 2 分量），最右下栅格的右下角落（不是栅格中心）为 Z(1,1)。如 P(1/2,1/3) 表示图像中一半高度上、水平三等分偏左的那个分点
+ “栅格 [a,b] 的中心值（即原图像和新图像的栅格数值）用 v[a,b] 表示
+ “栅格 [a,b] 的中心坐标、h 坐标、w 坐标”分别用 c[a,b]，h[a,b]，w[a,b] 表示，且有 c[a,b] = (h[a,b],w[a,b])
+ 原始图像相关变量用下标 1 标示，新图像相关变量用下标 2 标示。如 $h_{1}, w_{1}, h_{2}, w_{2}$ 分别表示原图像高度、原图像宽度、新图像高度、新图像宽度
+ $i=D_{1}^{n}$ 表示 $i=1,2,...,n$，就是一个缩略写法而已
+ $\lfloor a \rfloor, \lceil a \rceil, \{a\}$ 分别表示向下取整、向上取整和取小数部分（$\{a\} = a - \lfloor a \rfloor$）

### HALF_PIXEL (TensorRT7 align_corners=False)
+ 等效于 pyTorch(align_corners=False)
+ 原始图像的**四个角落（而不是四个中心点）**与新图像的四个角落重合，以两图像左上角角落为原点建立坐标系，然后计算新图像的各栅格中心点在原图像坐标系中的坐标，并进行双线性插值
    - 原图像上 $c_{1}\left[i,j\right] = \left(\frac{1}{2h_{1}}+\frac{i}{h_{1}},\frac{1}{2w_{1}}+\frac{j}{w_{1}}\right),i=D_{0}^{h_{1}-1},j=D_{0}^{w_{1}-1}$
    - 新图像上 $c_{2}\left[i,j\right] = \left(\frac{1}{2h_{1}}+\frac{i}{h_{2}},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\right),i=D_{0}^{h_{2}-1},j=D_{0}^{w_{2}-1}$
    - 即半格高度或宽度表示从原点到 (0,0) 栅格中心的偏移，相邻栅格坐标差值等于栅格宽度
    - 求新图像上栅格 [a,b] 中心值（即 $v_{2}[a,b]$,插值计算结果）时要找到原图像上的四个栅格用于插值，记用来插值的四个栅格中左上角那个栅格为 [p,q] 则有：$h_{1}[p,q] \le h_{2}[a,b] < h_{1}[p+1,q]，w_{1}[p,q] \le w_{2}[a,b] < w_{1}[p,q+1]$
    - 记 $\alpha = \frac{h_{1}}{h_{2}}\left(a+\frac{1}{2}\right)-\frac{1}{2},\beta = \frac{w_{1}}{w_{2}}\left(a+\frac{1}{2}\right)-\frac{1}{2}$，则上面两个不等式的解可以写成：$p = \lfloor \alpha \rfloor, q = \lfloor \beta \rfloor$
    - 于是插值结果写作：
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

+ 注意，当做上采样（Upsampling）时，新图像的最外圈栅格中心点坐标可能位于原图像最外圈栅格中心点之外，
    - TensorRT<7.2，最外圈栅格中心不做特殊处理，依然按照公式计算（结果如下图示中的 step1 所示）
    - TensorRT>=7.2，最外圈栅格中心会单独“拉回”原图像的最外圈栅格中心构成的矩形之内（结果如下图示中的 step2 所示），此时新图像中最外圈栅格中心的间距与内部栅格中心之间的距离不相等

+ 图示
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:70%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:70%;" />
</div>
<div align="center" >
<img src="./ResizeLayer-HALF_PIXEL-Step1.png" alt="ResizeLayer-HALF_PIXEL-Step1" style="zoom:70%;" />
<img src="./ResizeLayer-HALF_PIXEL-Step2.png" alt="ResizeLayer-HALF_PIXEL-Step2" style="zoom:70%;" />
</div>

### ALIGN_CORNERS (TensorRT align_corners=True)
+ 等效于 pyTorch(align_corners=True)
+ 原始图像**四个角栅格的中心点**与新图像的四个角栅格中心点对齐，以新图像左上角角落为原点建立坐标系，然后计算新图像的各栅格中心点在原图像坐标系中的坐标，并进行双线性插值
    - 原图像上 $c_{1}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{2}}\cdot\frac{h_{2}-1}{h_{1}-1},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\cdot\frac{w_{2}-1}{w_{1}-1}\right),i=D_{0}^{h_{1}-1},j=D_{0}^{w_{1}-1}$
    - 新图像上 $c_{2}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{2}},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\right),i=D_{0}^{h_{2}-1},j=D_{0}^{w_{2}-1}$
    - 即半格偏移按新图像计算，另外由于原图像边长发生了缩放，需要多乘一个因子，可以考虑 i=0 和 i=h_{1}-1 的情况来理解
    - 计算过程与前面相同，记 $\alpha = \frac{h_{1}-1}{h_{2}-1}a,\beta = \frac{w_{1}-1}{h_{2}-1}b$，则上面两个不等式的解可以写成：$p = \lfloor \alpha \rfloor, q = \lfloor \beta \rfloor$
    - 于是插值结果依然写作：
$$
\begin{aligned}
v_{2}[a,b] &=
\left(1-\{\alpha\}\right)\left(1-\{\beta\}\right) v_{1}[p,q] + \{\alpha\}\left(1-\{\beta\}\right) v_{1}[p+1,q] + \\
&\quad\;\left(1-\{\alpha\}\right)\{\beta\} v_{1}[p,q+1] + \{\alpha\}\{\beta\} v_{1}[p+1,q+1]
\end{aligned}
$$

+ 图示
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:70%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:70%;" />
<img src="./ResizeLayer-ALIGN_CORNERS.png" alt="ResizeLayer-ALIGN_CORNERS" style="zoom:70%;" />
</div>

### ASYMMETRIC (TensorRT<7 align_corners=False)
+ 原始图像与新图像边长均归一化为 1，然后将原始图像**左上角栅格的中心点**与新图像左上角栅格的中心点对齐，不做进一步的缩放处理，以新图像左上角角落为原点建立坐标系，然后计算新图像的各栅格中心点在原图像坐标系中的坐标，并进行双线性插值
    - 原图像上 $c_{1}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{1}},\frac{1}{2w_{2}}+\frac{j}{w_{1}}\right),i=D_{0}^{h_{1}-1},j=D_{0}^{w_{1}-1}$
    - 新图像上 $c_{2}\left[i,j\right] = \left(\frac{1}{2h_{2}}+\frac{i}{h_{2}},\frac{1}{2w_{2}}+\frac{j}{w_{2}}\right),i=D_{0}^{h_{2}-1},j=D_{0}^{w_{2}-1}$
    - 即半格偏移按新图像计算，但相邻栅格距离仍然按照原图像和新图像各自的栅格数计算
    - 计算过程与前面相同，记 $\alpha = \frac{h_{1}}{h_{2}}a,\beta = \frac{w_{1}}{h_{2}}b$，则上面两个不等式的解可以写成：$p = \lfloor \alpha \rfloor, q = \lfloor \beta \rfloor$
    - 于是插值结果依然写作：
$$
\begin{aligned}
v_{2}[a,b] &=
\left(1-\{\alpha\}\right)\left(1-\{\beta\}\right) v_{1}[p,q] + \{\alpha\}\left(1-\{\beta\}\right) v_{1}[p+1,q] + \\
&\quad\;\left(1-\{\alpha\}\right)\{\beta\} v_{1}[p,q+1] + \{\alpha\}\{\beta\} v_{1}[p+1,q+1]
\end{aligned}
$$

+ 注意，当做上采样（Upsampling）时，新图像的最外圈栅格中心点坐标可能位于原图像最外圈栅格中心点之外，
    - TensorRT<7.2，最外圈栅格中心不做特殊处理，依然按照公式计算（结果如下图示中的 step1 所示）
    - TensorRT>=7.2，最外圈栅格中心会单独“拉回”原图像的最外圈栅格中心构成的矩形之内（结果如下图示中的 step2 所示），此时新图像中最外圈栅格中心的间距与内部栅格中心之间的距离不相等
    -
+ 图示
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:70%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:70%;" />
</div>
<div align="center" >
<img src="./ResizeLayer-ASYMMETRIC-Step1.png" alt="ResizeLayer-ASYMMETRIC-Step1" style="zoom:70%;" />
<img src="./ResizeLayer-ASYMMETRIC-Step2.png" alt="ResizeLayer-ASYMMETRIC-Step2" style="zoom:70%;" />
</div>
