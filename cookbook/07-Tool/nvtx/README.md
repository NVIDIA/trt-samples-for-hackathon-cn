# nvtx

+ Use NVIDIA®Tools Extension SDK to add mark in timeline of Nsight systems.

+ Requires `nvtx >= 0.2.16`.

+ Steps to run.

```bash
nsys profile \
    -f true \
    -o py \
    python3 main.py

make
nsys profile \
    -f true \
    -o cpp \
    ./main.exe
```

+ Without a profiler attached, `nvtx.get_domain()` returns a dummy object and every
  call returns immediately, so the annotations can be left in production code. Run
  `python3 main.py` on its own and each case prints `DummyDomain` / `DummyCounter`
  to say so.

## Where the events land

Ranges show up in `nvtx_sum`, prefixed by their domain:

```bash
nsys stats --report nvtx_sum py.nsys-rep
```

```
 Instances    Style                       Range
        30  PushPop   NVTX-cookbook:enqueue              # annotate / push_range / payload
        10  StartEnd  NVTX-cookbook:enqueue              # start_range
        10  PushPop   NVTX-cookbook:enqueue-fast         # Domain object + reused EventAttributes
        10  PushPop   NVTX-cookbook:preprocess           # decorator, message from the function name
        10  PushPop   NVTX-cookbook:postprocess-renamed  # decorator with an explicit message
        10  PushPop   nvtx.py:main.py:59(infer_once)     # nvtx.Profile
```

**Counters do not appear in `nvtx_sum`.** They are plotted as their own rows in the
Nsight Systems GUI, and in the exported database they live in `GENERIC_EVENTS`:

```python
import sqlite3
c = sqlite3.connect("py.sqlite")  # nsys stats writes this next to the .nsys-rep
print(c.execute("select count(*) from GENERIC_EVENTS").fetchone())
```

+ Check full color table

```python
import matplotlib.colors as mcolors

print("BASE_COLORS")
for name, rgb in mcolors.BASE_COLORS.items():
    print(f"| {name:2s} | {rgb} |")

print("CSS4_COLORS")
for name, hexv in mcolors.CSS4_COLORS.items():
    print(f"| {name:25s} | {hexv} |")

print("TABLEAU_COLORS")
for name, hexv in mcolors.TABLEAU_COLORS.items():
    print(f"| {name:25s} | {hexv} |")

print("XKCD_COLORS")
for name, hexv in mcolors.XKCD_COLORS.items():
    print(f"| {name:25s} | {hexv} |")
```

+ Available colors

+ 8 basic colors

|  Name   | Abbreviation |   RGB   |                 Example                  |
| :-----: | :----------: | :-----: | :--------------------------------------: |
|  blue   |      b       | #0000ff | $\fcolorbox{gray}{#0000ff}{\phantom{x}}$ |
|  green  |      g       | #00ff00 | $\fcolorbox{gray}{#00ff00}{\phantom{x}}$ |
|   red   |      r       | #ff0000 | $\fcolorbox{gray}{#ff0000}{\phantom{x}}$ |
|  cyan   |      c       | #00ffff | $\fcolorbox{gray}{#00ffff}{\phantom{x}}$ |
| magenta |      m       | #ff00ff | $\fcolorbox{gray}{#ff00ff}{\phantom{x}}$ |
| yellow  |      y       | #ffff00 | $\fcolorbox{gray}{#ffff00}{\phantom{x}}$ |
|  black  |      k       | #000000 | $\fcolorbox{gray}{#000000}{\phantom{x}}$ |
|  white  |      w       | #ffffff | $\fcolorbox{gray}{#ffffff}{\phantom{x}}$ |

+ Tab10 colors

|    Name    |   RGB   |                 Example                  |
| :--------: | :-----: | :--------------------------------------: |
|  tab:blue  | #1f77b4 | $\fcolorbox{gray}{#1f77b4}{\phantom{x}}$ |
| tab:orange | #ff7f0e | $\fcolorbox{gray}{#ff7f0e}{\phantom{x}}$ |
| tab:green  | #2ca02c | $\fcolorbox{gray}{#2ca02c}{\phantom{x}}$ |
|  tab:red   | #d62728 | $\fcolorbox{gray}{#d62728}{\phantom{x}}$ |
| tab:purple | #9467bd | $\fcolorbox{gray}{#9467bd}{\phantom{x}}$ |
| tab:brown  | #8c564b | $\fcolorbox{gray}{#8c564b}{\phantom{x}}$ |
|  tab:pink  | #e377c2 | $\fcolorbox{gray}{#e377c2}{\phantom{x}}$ |
|  tab:gray  | #7f7f7f | $\fcolorbox{gray}{#7f7f7f}{\phantom{x}}$ |
| tab:olive  | #bcbd22 | $\fcolorbox{gray}{#bcbd22}{\phantom{x}}$ |
|  tab:cyan  | #17becf | $\fcolorbox{gray}{#17becf}{\phantom{x}}$ |


+ CSS/X11 147 colors

|         Name         |   RGB   |                 Example                  |
| :------------------: | :-----: | :--------------------------------------: |
|      aliceblue       | #F0F8FF | $\fcolorbox{gray}{#F0F8FF}{\phantom{x}}$ |
|     antiquewhite     | #FAEBD7 | $\fcolorbox{gray}{#FAEBD7}{\phantom{x}}$ |
|         aqua         | #00FFFF | $\fcolorbox{gray}{#00FFFF}{\phantom{x}}$ |
|      aquamarine      | #7FFFD4 | $\fcolorbox{gray}{#7FFFD4}{\phantom{x}}$ |
|        azure         | #F0FFFF | $\fcolorbox{gray}{#F0FFFF}{\phantom{x}}$ |
|        beige         | #F5F5DC | $\fcolorbox{gray}{#F5F5DC}{\phantom{x}}$ |
|        bisque        | #FFE4C4 | $\fcolorbox{gray}{#FFE4C4}{\phantom{x}}$ |
|        black         | #000000 | $\fcolorbox{gray}{#000000}{\phantom{x}}$ |
|    blanchedalmond    | #FFEBCD | $\fcolorbox{gray}{#FFEBCD}{\phantom{x}}$ |
|         blue         | #0000FF | $\fcolorbox{gray}{#0000FF}{\phantom{x}}$ |
|      blueviolet      | #8A2BE2 | $\fcolorbox{gray}{#8A2BE2}{\phantom{x}}$ |
|        brown         | #A52A2A | $\fcolorbox{gray}{#A52A2A}{\phantom{x}}$ |
|      burlywood       | #DEB887 | $\fcolorbox{gray}{#DEB887}{\phantom{x}}$ |
|      cadetblue       | #5F9EA0 | $\fcolorbox{gray}{#5F9EA0}{\phantom{x}}$ |
|      chartreuse      | #7FFF00 | $\fcolorbox{gray}{#7FFF00}{\phantom{x}}$ |
|      chocolate       | #D2691E | $\fcolorbox{gray}{#D2691E}{\phantom{x}}$ |
|        coral         | #FF7F50 | $\fcolorbox{gray}{#FF7F50}{\phantom{x}}$ |
|    cornflowerblue    | #6495ED | $\fcolorbox{gray}{#6495ED}{\phantom{x}}$ |
|       cornsilk       | #FFF8DC | $\fcolorbox{gray}{#FFF8DC}{\phantom{x}}$ |
|       crimson        | #DC143C | $\fcolorbox{gray}{#DC143C}{\phantom{x}}$ |
|         cyan         | #00FFFF | $\fcolorbox{gray}{#00FFFF}{\phantom{x}}$ |
|       darkblue       | #00008B | $\fcolorbox{gray}{#00008B}{\phantom{x}}$ |
|       darkcyan       | #008B8B | $\fcolorbox{gray}{#008B8B}{\phantom{x}}$ |
|    darkgoldenrod     | #B8860B | $\fcolorbox{gray}{#B8860B}{\phantom{x}}$ |
|       darkgray       | #A9A9A9 | $\fcolorbox{gray}{#A9A9A9}{\phantom{x}}$ |
|      darkgreen       | #006400 | $\fcolorbox{gray}{#006400}{\phantom{x}}$ |
|       darkgrey       | #A9A9A9 | $\fcolorbox{gray}{#A9A9A9}{\phantom{x}}$ |
|      darkkhaki       | #BDB76B | $\fcolorbox{gray}{#BDB76B}{\phantom{x}}$ |
|     darkmagenta      | #8B008B | $\fcolorbox{gray}{#8B008B}{\phantom{x}}$ |
|    darkolivegreen    | #556B2F | $\fcolorbox{gray}{#556B2F}{\phantom{x}}$ |
|      darkorange      | #FF8C00 | $\fcolorbox{gray}{#FF8C00}{\phantom{x}}$ |
|      darkorchid      | #9932CC | $\fcolorbox{gray}{#9932CC}{\phantom{x}}$ |
|       darkred        | #8B0000 | $\fcolorbox{gray}{#8B0000}{\phantom{x}}$ |
|      darksalmon      | #E9967A | $\fcolorbox{gray}{#E9967A}{\phantom{x}}$ |
|     darkseagreen     | #8FBC8F | $\fcolorbox{gray}{#8FBC8F}{\phantom{x}}$ |
|    darkslateblue     | #483D8B | $\fcolorbox{gray}{#483D8B}{\phantom{x}}$ |
|    darkslategray     | #2F4F4F | $\fcolorbox{gray}{#2F4F4F}{\phantom{x}}$ |
|    darkslategrey     | #2F4F4F | $\fcolorbox{gray}{#2F4F4F}{\phantom{x}}$ |
|    darkturquoise     | #00CED1 | $\fcolorbox{gray}{#00CED1}{\phantom{x}}$ |
|      darkviolet      | #9400D3 | $\fcolorbox{gray}{#9400D3}{\phantom{x}}$ |
|       deeppink       | #FF1493 | $\fcolorbox{gray}{#FF1493}{\phantom{x}}$ |
|     deepskyblue      | #00BFFF | $\fcolorbox{gray}{#00BFFF}{\phantom{x}}$ |
|       dimgray        | #696969 | $\fcolorbox{gray}{#696969}{\phantom{x}}$ |
|       dimgrey        | #696969 | $\fcolorbox{gray}{#696969}{\phantom{x}}$ |
|      dodgerblue      | #1E90FF | $\fcolorbox{gray}{#1E90FF}{\phantom{x}}$ |
|      firebrick       | #B22222 | $\fcolorbox{gray}{#B22222}{\phantom{x}}$ |
|     floralwhite      | #FFFAF0 | $\fcolorbox{gray}{#FFFAF0}{\phantom{x}}$ |
|     forestgreen      | #228B22 | $\fcolorbox{gray}{#228B22}{\phantom{x}}$ |
|       fuchsia        | #FF00FF | $\fcolorbox{gray}{#FF00FF}{\phantom{x}}$ |
|      gainsboro       | #DCDCDC | $\fcolorbox{gray}{#DCDCDC}{\phantom{x}}$ |
|      ghostwhite      | #F8F8FF | $\fcolorbox{gray}{#F8F8FF}{\phantom{x}}$ |
|         gold         | #FFD700 | $\fcolorbox{gray}{#FFD700}{\phantom{x}}$ |
|      goldenrod       | #DAA520 | $\fcolorbox{gray}{#DAA520}{\phantom{x}}$ |
|         gray         | #808080 | $\fcolorbox{gray}{#808080}{\phantom{x}}$ |
|        green         | #008000 | $\fcolorbox{gray}{#008000}{\phantom{x}}$ |
|     greenyellow      | #ADFF2F | $\fcolorbox{gray}{#ADFF2F}{\phantom{x}}$ |
|         grey         | #808080 | $\fcolorbox{gray}{#808080}{\phantom{x}}$ |
|       honeydew       | #F0FFF0 | $\fcolorbox{gray}{#F0FFF0}{\phantom{x}}$ |
|       hotpink        | #FF69B4 | $\fcolorbox{gray}{#FF69B4}{\phantom{x}}$ |
|      indianred       | #CD5C5C | $\fcolorbox{gray}{#CD5C5C}{\phantom{x}}$ |
|        indigo        | #4B0082 | $\fcolorbox{gray}{#4B0082}{\phantom{x}}$ |
|        ivory         | #FFFFF0 | $\fcolorbox{gray}{#FFFFF0}{\phantom{x}}$ |
|        khaki         | #F0E68C | $\fcolorbox{gray}{#F0E68C}{\phantom{x}}$ |
|       lavender       | #E6E6FA | $\fcolorbox{gray}{#E6E6FA}{\phantom{x}}$ |
|    lavenderblush     | #FFF0F5 | $\fcolorbox{gray}{#FFF0F5}{\phantom{x}}$ |
|      lawngreen       | #7CFC00 | $\fcolorbox{gray}{#7CFC00}{\phantom{x}}$ |
|     lemonchiffon     | #FFFACD | $\fcolorbox{gray}{#FFFACD}{\phantom{x}}$ |
|      lightblue       | #ADD8E6 | $\fcolorbox{gray}{#ADD8E6}{\phantom{x}}$ |
|      lightcoral      | #F08080 | $\fcolorbox{gray}{#F08080}{\phantom{x}}$ |
|      lightcyan       | #E0FFFF | $\fcolorbox{gray}{#E0FFFF}{\phantom{x}}$ |
| lightgoldenrodyellow | #FAFAD2 | $\fcolorbox{gray}{#FAFAD2}{\phantom{x}}$ |
|      lightgray       | #D3D3D3 | $\fcolorbox{gray}{#D3D3D3}{\phantom{x}}$ |
|      lightgreen      | #90EE90 | $\fcolorbox{gray}{#90EE90}{\phantom{x}}$ |
|      lightgrey       | #D3D3D3 | $\fcolorbox{gray}{#D3D3D3}{\phantom{x}}$ |
|      lightpink       | #FFB6C1 | $\fcolorbox{gray}{#FFB6C1}{\phantom{x}}$ |
|     lightsalmon      | #FFA07A | $\fcolorbox{gray}{#FFA07A}{\phantom{x}}$ |
|    lightseagreen     | #20B2AA | $\fcolorbox{gray}{#20B2AA}{\phantom{x}}$ |
|     lightskyblue     | #87CEFA | $\fcolorbox{gray}{#87CEFA}{\phantom{x}}$ |
|    lightslategray    | #778899 | $\fcolorbox{gray}{#778899}{\phantom{x}}$ |
|    lightslategrey    | #778899 | $\fcolorbox{gray}{#778899}{\phantom{x}}$ |
|    lightsteelblue    | #B0C4DE | $\fcolorbox{gray}{#B0C4DE}{\phantom{x}}$ |
|     lightyellow      | #FFFFE0 | $\fcolorbox{gray}{#FFFFE0}{\phantom{x}}$ |
|         lime         | #00FF00 | $\fcolorbox{gray}{#00FF00}{\phantom{x}}$ |
|      limegreen       | #32CD32 | $\fcolorbox{gray}{#32CD32}{\phantom{x}}$ |
|        linen         | #FAF0E6 | $\fcolorbox{gray}{#FAF0E6}{\phantom{x}}$ |
|       magenta        | #FF00FF | $\fcolorbox{gray}{#FF00FF}{\phantom{x}}$ |
|        maroon        | #800000 | $\fcolorbox{gray}{#800000}{\phantom{x}}$ |
|   mediumaquamarine   | #66CDAA | $\fcolorbox{gray}{#66CDAA}{\phantom{x}}$ |
|      mediumblue      | #0000CD | $\fcolorbox{gray}{#0000CD}{\phantom{x}}$ |
|     mediumorchid     | #BA55D3 | $\fcolorbox{gray}{#BA55D3}{\phantom{x}}$ |
|     mediumpurple     | #9370DB | $\fcolorbox{gray}{#9370DB}{\phantom{x}}$ |
|    mediumseagreen    | #3CB371 | $\fcolorbox{gray}{#3CB371}{\phantom{x}}$ |
|   mediumslateblue    | #7B68EE | $\fcolorbox{gray}{#7B68EE}{\phantom{x}}$ |
|  mediumspringgreen   | #00FA9A | $\fcolorbox{gray}{#00FA9A}{\phantom{x}}$ |
|   mediumturquoise    | #48D1CC | $\fcolorbox{gray}{#48D1CC}{\phantom{x}}$ |
|   mediumvioletred    | #C71585 | $\fcolorbox{gray}{#C71585}{\phantom{x}}$ |
|     midnightblue     | #191970 | $\fcolorbox{gray}{#191970}{\phantom{x}}$ |
|      mintcream       | #F5FFFA | $\fcolorbox{gray}{#F5FFFA}{\phantom{x}}$ |
|      mistyrose       | #FFE4E1 | $\fcolorbox{gray}{#FFE4E1}{\phantom{x}}$ |
|       moccasin       | #FFE4B5 | $\fcolorbox{gray}{#FFE4B5}{\phantom{x}}$ |
|     navajowhite      | #FFDEAD | $\fcolorbox{gray}{#FFDEAD}{\phantom{x}}$ |
|         navy         | #000080 | $\fcolorbox{gray}{#000080}{\phantom{x}}$ |
|       oldlace        | #FDF5E6 | $\fcolorbox{gray}{#FDF5E6}{\phantom{x}}$ |
|        olive         | #808000 | $\fcolorbox{gray}{#808000}{\phantom{x}}$ |
|      olivedrab       | #6B8E23 | $\fcolorbox{gray}{#6B8E23}{\phantom{x}}$ |
|        orange        | #FFA500 | $\fcolorbox{gray}{#FFA500}{\phantom{x}}$ |
|      orangered       | #FF4500 | $\fcolorbox{gray}{#FF4500}{\phantom{x}}$ |
|        orchid        | #DA70D6 | $\fcolorbox{gray}{#DA70D6}{\phantom{x}}$ |
|    palegoldenrod     | #EEE8AA | $\fcolorbox{gray}{#EEE8AA}{\phantom{x}}$ |
|      palegreen       | #98FB98 | $\fcolorbox{gray}{#98FB98}{\phantom{x}}$ |
|    paleturquoise     | #AFEEEE | $\fcolorbox{gray}{#AFEEEE}{\phantom{x}}$ |
|    palevioletred     | #DB7093 | $\fcolorbox{gray}{#DB7093}{\phantom{x}}$ |
|      papayawhip      | #FFEFD5 | $\fcolorbox{gray}{#FFEFD5}{\phantom{x}}$ |
|      peachpuff       | #FFDAB9 | $\fcolorbox{gray}{#FFDAB9}{\phantom{x}}$ |
|         peru         | #CD853F | $\fcolorbox{gray}{#CD853F}{\phantom{x}}$ |
|         pink         | #FFC0CB | $\fcolorbox{gray}{#FFC0CB}{\phantom{x}}$ |
|         plum         | #DDA0DD | $\fcolorbox{gray}{#DDA0DD}{\phantom{x}}$ |
|      powderblue      | #B0E0E6 | $\fcolorbox{gray}{#B0E0E6}{\phantom{x}}$ |
|        purple        | #800080 | $\fcolorbox{gray}{#800080}{\phantom{x}}$ |
|    rebeccapurple     | #663399 | $\fcolorbox{gray}{#663399}{\phantom{x}}$ |
|         red          | #FF0000 | $\fcolorbox{gray}{#FF0000}{\phantom{x}}$ |
|      rosybrown       | #BC8F8F | $\fcolorbox{gray}{#BC8F8F}{\phantom{x}}$ |
|      royalblue       | #4169E1 | $\fcolorbox{gray}{#4169E1}{\phantom{x}}$ |
|     saddlebrown      | #8B4513 | $\fcolorbox{gray}{#8B4513}{\phantom{x}}$ |
|        salmon        | #FA8072 | $\fcolorbox{gray}{#FA8072}{\phantom{x}}$ |
|      sandybrown      | #F4A460 | $\fcolorbox{gray}{#F4A460}{\phantom{x}}$ |
|       seagreen       | #2E8B57 | $\fcolorbox{gray}{#2E8B57}{\phantom{x}}$ |
|       seashell       | #FFF5EE | $\fcolorbox{gray}{#FFF5EE}{\phantom{x}}$ |
|        sienna        | #A0522D | $\fcolorbox{gray}{#A0522D}{\phantom{x}}$ |
|        silver        | #C0C0C0 | $\fcolorbox{gray}{#C0C0C0}{\phantom{x}}$ |
|       skyblue        | #87CEEB | $\fcolorbox{gray}{#87CEEB}{\phantom{x}}$ |
|      slateblue       | #6A5ACD | $\fcolorbox{gray}{#6A5ACD}{\phantom{x}}$ |
|      slategray       | #708090 | $\fcolorbox{gray}{#708090}{\phantom{x}}$ |
|      slategrey       | #708090 | $\fcolorbox{gray}{#708090}{\phantom{x}}$ |
|         snow         | #FFFAFA | $\fcolorbox{gray}{#FFFAFA}{\phantom{x}}$ |
|     springgreen      | #00FF7F | $\fcolorbox{gray}{#00FF7F}{\phantom{x}}$ |
|      steelblue       | #4682B4 | $\fcolorbox{gray}{#4682B4}{\phantom{x}}$ |
|         tan          | #D2B48C | $\fcolorbox{gray}{#D2B48C}{\phantom{x}}$ |
|         teal         | #008080 | $\fcolorbox{gray}{#008080}{\phantom{x}}$ |
|       thistle        | #D8BFD8 | $\fcolorbox{gray}{#D8BFD8}{\phantom{x}}$ |
|        tomato        | #FF6347 | $\fcolorbox{gray}{#FF6347}{\phantom{x}}$ |
|      turquoise       | #40E0D0 | $\fcolorbox{gray}{#40E0D0}{\phantom{x}}$ |
|        violet        | #EE82EE | $\fcolorbox{gray}{#EE82EE}{\phantom{x}}$ |
|        wheat         | #F5DEB3 | $\fcolorbox{gray}{#F5DEB3}{\phantom{x}}$ |
|        white         | #FFFFFF | $\fcolorbox{gray}{#FFFFFF}{\phantom{x}}$ |
|      whitesmoke      | #F5F5F5 | $\fcolorbox{gray}{#F5F5F5}{\phantom{x}}$ |
|        yellow        | #FFFF00 | $\fcolorbox{gray}{#FFFF00}{\phantom{x}}$ |
|     yellowgreen      | #9ACD32 | $\fcolorbox{gray}{#9ACD32}{\phantom{x}}$ |

+ xkcd 954 colors (https://xkcd.com/color/rgb/)

|              Name               |   RGB   |                 Example                  |
| :-----------------------------: | :-----: | :--------------------------------------: |
|        xkcd:cloudy blue         | #acc2d9 | $\fcolorbox{gray}{#acc2d9}{\phantom{x}}$ |
|     xkcd:dark pastel green      | #56ae57 | $\fcolorbox{gray}{#56ae57}{\phantom{x}}$ |
|            xkcd:dust            | #b2996e | $\fcolorbox{gray}{#b2996e}{\phantom{x}}$ |
|       xkcd:electric lime        | #a8ff04 | $\fcolorbox{gray}{#a8ff04}{\phantom{x}}$ |
|        xkcd:fresh green         | #69d84f | $\fcolorbox{gray}{#69d84f}{\phantom{x}}$ |
|       xkcd:light eggplant       | #894585 | $\fcolorbox{gray}{#894585}{\phantom{x}}$ |
|        xkcd:nasty green         | #70b23f | $\fcolorbox{gray}{#70b23f}{\phantom{x}}$ |
|     xkcd:really light blue      | #d4ffff | $\fcolorbox{gray}{#d4ffff}{\phantom{x}}$ |
|            xkcd:tea             | #65ab7c | $\fcolorbox{gray}{#65ab7c}{\phantom{x}}$ |
|        xkcd:warm purple         | #952e8f | $\fcolorbox{gray}{#952e8f}{\phantom{x}}$ |
|       xkcd:yellowish tan        | #fcfc81 | $\fcolorbox{gray}{#fcfc81}{\phantom{x}}$ |
|           xkcd:cement           | #a5a391 | $\fcolorbox{gray}{#a5a391}{\phantom{x}}$ |
|      xkcd:dark grass green      | #388004 | $\fcolorbox{gray}{#388004}{\phantom{x}}$ |
|         xkcd:dusty teal         | #4c9085 | $\fcolorbox{gray}{#4c9085}{\phantom{x}}$ |
|         xkcd:grey teal          | #5e9b8a | $\fcolorbox{gray}{#5e9b8a}{\phantom{x}}$ |
|    xkcd:macaroni and cheese     | #efb435 | $\fcolorbox{gray}{#efb435}{\phantom{x}}$ |
|        xkcd:pinkish tan         | #d99b82 | $\fcolorbox{gray}{#d99b82}{\phantom{x}}$ |
|           xkcd:spruce           | #0a5f38 | $\fcolorbox{gray}{#0a5f38}{\phantom{x}}$ |
|        xkcd:strong blue         | #0c06f7 | $\fcolorbox{gray}{#0c06f7}{\phantom{x}}$ |
|        xkcd:toxic green         | #61de2a | $\fcolorbox{gray}{#61de2a}{\phantom{x}}$ |
|        xkcd:windows blue        | #3778bf | $\fcolorbox{gray}{#3778bf}{\phantom{x}}$ |
|         xkcd:blue blue          | #2242c7 | $\fcolorbox{gray}{#2242c7}{\phantom{x}}$ |
| xkcd:blue with a hint of purple | #533cc6 | $\fcolorbox{gray}{#533cc6}{\phantom{x}}$ |
|           xkcd:booger           | #9bb53c | $\fcolorbox{gray}{#9bb53c}{\phantom{x}}$ |
|      xkcd:bright sea green      | #05ffa6 | $\fcolorbox{gray}{#05ffa6}{\phantom{x}}$ |
|      xkcd:dark green blue       | #1f6357 | $\fcolorbox{gray}{#1f6357}{\phantom{x}}$ |
|       xkcd:deep turquoise       | #017374 | $\fcolorbox{gray}{#017374}{\phantom{x}}$ |
|         xkcd:green teal         | #0cb577 | $\fcolorbox{gray}{#0cb577}{\phantom{x}}$ |
|        xkcd:strong pink         | #ff0789 | $\fcolorbox{gray}{#ff0789}{\phantom{x}}$ |
|           xkcd:bland            | #afa88b | $\fcolorbox{gray}{#afa88b}{\phantom{x}}$ |
|         xkcd:deep aqua          | #08787f | $\fcolorbox{gray}{#08787f}{\phantom{x}}$ |
|       xkcd:lavender pink        | #dd85d7 | $\fcolorbox{gray}{#dd85d7}{\phantom{x}}$ |
|      xkcd:light moss green      | #a6c875 | $\fcolorbox{gray}{#a6c875}{\phantom{x}}$ |
|    xkcd:light seafoam green     | #a7ffb5 | $\fcolorbox{gray}{#a7ffb5}{\phantom{x}}$ |
|        xkcd:olive yellow        | #c2b709 | $\fcolorbox{gray}{#c2b709}{\phantom{x}}$ |
|          xkcd:pig pink          | #e78ea5 | $\fcolorbox{gray}{#e78ea5}{\phantom{x}}$ |
|         xkcd:deep lilac         | #966ebd | $\fcolorbox{gray}{#966ebd}{\phantom{x}}$ |
|           xkcd:desert           | #ccad60 | $\fcolorbox{gray}{#ccad60}{\phantom{x}}$ |
|       xkcd:dusty lavender       | #ac86a8 | $\fcolorbox{gray}{#ac86a8}{\phantom{x}}$ |
|        xkcd:purpley grey        | #947e94 | $\fcolorbox{gray}{#947e94}{\phantom{x}}$ |
|           xkcd:purply           | #983fb2 | $\fcolorbox{gray}{#983fb2}{\phantom{x}}$ |
|         xkcd:candy pink         | #ff63e9 | $\fcolorbox{gray}{#ff63e9}{\phantom{x}}$ |
|     xkcd:light pastel green     | #b2fba5 | $\fcolorbox{gray}{#b2fba5}{\phantom{x}}$ |
|        xkcd:boring green        | #63b365 | $\fcolorbox{gray}{#63b365}{\phantom{x}}$ |
|         xkcd:kiwi green         | #8ee53f | $\fcolorbox{gray}{#8ee53f}{\phantom{x}}$ |
|      xkcd:light grey green      | #b7e1a1 | $\fcolorbox{gray}{#b7e1a1}{\phantom{x}}$ |
|        xkcd:orange pink         | #ff6f52 | $\fcolorbox{gray}{#ff6f52}{\phantom{x}}$ |
|         xkcd:tea green          | #bdf8a3 | $\fcolorbox{gray}{#bdf8a3}{\phantom{x}}$ |
|      xkcd:very light brown      | #d3b683 | $\fcolorbox{gray}{#d3b683}{\phantom{x}}$ |
|         xkcd:egg shell          | #fffcc4 | $\fcolorbox{gray}{#fffcc4}{\phantom{x}}$ |
|      xkcd:eggplant purple       | #430541 | $\fcolorbox{gray}{#430541}{\phantom{x}}$ |
|        xkcd:powder pink         | #ffb2d0 | $\fcolorbox{gray}{#ffb2d0}{\phantom{x}}$ |
|        xkcd:reddish grey        | #997570 | $\fcolorbox{gray}{#997570}{\phantom{x}}$ |
|      xkcd:baby shit brown       | #ad900d | $\fcolorbox{gray}{#ad900d}{\phantom{x}}$ |
|           xkcd:liliac           | #c48efd | $\fcolorbox{gray}{#c48efd}{\phantom{x}}$ |
|        xkcd:stormy blue         | #507b9c | $\fcolorbox{gray}{#507b9c}{\phantom{x}}$ |
|         xkcd:ugly brown         | #7d7103 | $\fcolorbox{gray}{#7d7103}{\phantom{x}}$ |
|          xkcd:custard           | #fffd78 | $\fcolorbox{gray}{#fffd78}{\phantom{x}}$ |
|        xkcd:darkish pink        | #da467d | $\fcolorbox{gray}{#da467d}{\phantom{x}}$ |
|         xkcd:deep brown         | #410200 | $\fcolorbox{gray}{#410200}{\phantom{x}}$ |
|       xkcd:greenish beige       | #c9d179 | $\fcolorbox{gray}{#c9d179}{\phantom{x}}$ |
|          xkcd:manilla           | #fffa86 | $\fcolorbox{gray}{#fffa86}{\phantom{x}}$ |
|          xkcd:off blue          | #5684ae | $\fcolorbox{gray}{#5684ae}{\phantom{x}}$ |
|      xkcd:battleship grey       | #6b7c85 | $\fcolorbox{gray}{#6b7c85}{\phantom{x}}$ |
|        xkcd:browny green        | #6f6c0a | $\fcolorbox{gray}{#6f6c0a}{\phantom{x}}$ |
|           xkcd:bruise           | #7e4071 | $\fcolorbox{gray}{#7e4071}{\phantom{x}}$ |
|        xkcd:kelley green        | #009337 | $\fcolorbox{gray}{#009337}{\phantom{x}}$ |
|       xkcd:sickly yellow        | #d0e429 | $\fcolorbox{gray}{#d0e429}{\phantom{x}}$ |
|        xkcd:sunny yellow        | #fff917 | $\fcolorbox{gray}{#fff917}{\phantom{x}}$ |
|            xkcd:azul            | #1d5dec | $\fcolorbox{gray}{#1d5dec}{\phantom{x}}$ |
|         xkcd:darkgreen          | #054907 | $\fcolorbox{gray}{#054907}{\phantom{x}}$ |
|        xkcd:green/yellow        | #b5ce08 | $\fcolorbox{gray}{#b5ce08}{\phantom{x}}$ |
|           xkcd:lichen           | #8fb67b | $\fcolorbox{gray}{#8fb67b}{\phantom{x}}$ |
|     xkcd:light light green      | #c8ffb0 | $\fcolorbox{gray}{#c8ffb0}{\phantom{x}}$ |
|         xkcd:pale gold          | #fdde6c | $\fcolorbox{gray}{#fdde6c}{\phantom{x}}$ |
|         xkcd:sun yellow         | #ffdf22 | $\fcolorbox{gray}{#ffdf22}{\phantom{x}}$ |
|         xkcd:tan green          | #a9be70 | $\fcolorbox{gray}{#a9be70}{\phantom{x}}$ |
|           xkcd:burple           | #6832e3 | $\fcolorbox{gray}{#6832e3}{\phantom{x}}$ |
|        xkcd:butterscotch        | #fdb147 | $\fcolorbox{gray}{#fdb147}{\phantom{x}}$ |
|           xkcd:toupe            | #c7ac7d | $\fcolorbox{gray}{#c7ac7d}{\phantom{x}}$ |
|         xkcd:dark cream         | #fff39a | $\fcolorbox{gray}{#fff39a}{\phantom{x}}$ |
|         xkcd:indian red         | #850e04 | $\fcolorbox{gray}{#850e04}{\phantom{x}}$ |
|       xkcd:light lavendar       | #efc0fe | $\fcolorbox{gray}{#efc0fe}{\phantom{x}}$ |
|        xkcd:poison green        | #40fd14 | $\fcolorbox{gray}{#40fd14}{\phantom{x}}$ |
|      xkcd:baby puke green       | #b6c406 | $\fcolorbox{gray}{#b6c406}{\phantom{x}}$ |
|    xkcd:bright yellow green     | #9dff00 | $\fcolorbox{gray}{#9dff00}{\phantom{x}}$ |
|       xkcd:charcoal grey        | #3c4142 | $\fcolorbox{gray}{#3c4142}{\phantom{x}}$ |
|           xkcd:squash           | #f2ab15 | $\fcolorbox{gray}{#f2ab15}{\phantom{x}}$ |
|          xkcd:cinnamon          | #ac4f06 | $\fcolorbox{gray}{#ac4f06}{\phantom{x}}$ |
|      xkcd:light pea green       | #c4fe82 | $\fcolorbox{gray}{#c4fe82}{\phantom{x}}$ |
|     xkcd:radioactive green      | #2cfa1f | $\fcolorbox{gray}{#2cfa1f}{\phantom{x}}$ |
|         xkcd:raw sienna         | #9a6200 | $\fcolorbox{gray}{#9a6200}{\phantom{x}}$ |
|        xkcd:baby purple         | #ca9bf7 | $\fcolorbox{gray}{#ca9bf7}{\phantom{x}}$ |
|           xkcd:cocoa            | #875f42 | $\fcolorbox{gray}{#875f42}{\phantom{x}}$ |
|      xkcd:light royal blue      | #3a2efe | $\fcolorbox{gray}{#3a2efe}{\phantom{x}}$ |
|         xkcd:orangeish          | #fd8d49 | $\fcolorbox{gray}{#fd8d49}{\phantom{x}}$ |
|         xkcd:rust brown         | #8b3103 | $\fcolorbox{gray}{#8b3103}{\phantom{x}}$ |
|         xkcd:sand brown         | #cba560 | $\fcolorbox{gray}{#cba560}{\phantom{x}}$ |
|           xkcd:swamp            | #698339 | $\fcolorbox{gray}{#698339}{\phantom{x}}$ |
|       xkcd:tealish green        | #0cdc73 | $\fcolorbox{gray}{#0cdc73}{\phantom{x}}$ |
|        xkcd:burnt siena         | #b75203 | $\fcolorbox{gray}{#b75203}{\phantom{x}}$ |
|            xkcd:camo            | #7f8f4e | $\fcolorbox{gray}{#7f8f4e}{\phantom{x}}$ |
|         xkcd:dusk blue          | #26538d | $\fcolorbox{gray}{#26538d}{\phantom{x}}$ |
|            xkcd:fern            | #63a950 | $\fcolorbox{gray}{#63a950}{\phantom{x}}$ |
|          xkcd:old rose          | #c87f89 | $\fcolorbox{gray}{#c87f89}{\phantom{x}}$ |
|      xkcd:pale light green      | #b1fc99 | $\fcolorbox{gray}{#b1fc99}{\phantom{x}}$ |
|        xkcd:peachy pink         | #ff9a8a | $\fcolorbox{gray}{#ff9a8a}{\phantom{x}}$ |
|         xkcd:rosy pink          | #f6688e | $\fcolorbox{gray}{#f6688e}{\phantom{x}}$ |
|     xkcd:light bluish green     | #76fda8 | $\fcolorbox{gray}{#76fda8}{\phantom{x}}$ |
|     xkcd:light bright green     | #53fe5c | $\fcolorbox{gray}{#53fe5c}{\phantom{x}}$ |
|      xkcd:light neon green      | #4efd54 | $\fcolorbox{gray}{#4efd54}{\phantom{x}}$ |
|       xkcd:light seafoam        | #a0febf | $\fcolorbox{gray}{#a0febf}{\phantom{x}}$ |
|        xkcd:tiffany blue        | #7bf2da | $\fcolorbox{gray}{#7bf2da}{\phantom{x}}$ |
|      xkcd:washed out green      | #bcf5a6 | $\fcolorbox{gray}{#bcf5a6}{\phantom{x}}$ |
|       xkcd:browny orange        | #ca6b02 | $\fcolorbox{gray}{#ca6b02}{\phantom{x}}$ |
|         xkcd:nice blue          | #107ab0 | $\fcolorbox{gray}{#107ab0}{\phantom{x}}$ |
|          xkcd:sapphire          | #2138ab | $\fcolorbox{gray}{#2138ab}{\phantom{x}}$ |
|        xkcd:greyish teal        | #719f91 | $\fcolorbox{gray}{#719f91}{\phantom{x}}$ |
|       xkcd:orangey yellow       | #fdb915 | $\fcolorbox{gray}{#fdb915}{\phantom{x}}$ |
|         xkcd:parchment          | #fefcaf | $\fcolorbox{gray}{#fefcaf}{\phantom{x}}$ |
|           xkcd:straw            | #fcf679 | $\fcolorbox{gray}{#fcf679}{\phantom{x}}$ |
|      xkcd:very dark brown       | #1d0200 | $\fcolorbox{gray}{#1d0200}{\phantom{x}}$ |
|         xkcd:terracota          | #cb6843 | $\fcolorbox{gray}{#cb6843}{\phantom{x}}$ |
|         xkcd:ugly blue          | #31668a | $\fcolorbox{gray}{#31668a}{\phantom{x}}$ |
|         xkcd:clear blue         | #247afd | $\fcolorbox{gray}{#247afd}{\phantom{x}}$ |
|           xkcd:creme            | #ffffb6 | $\fcolorbox{gray}{#ffffb6}{\phantom{x}}$ |
|         xkcd:foam green         | #90fda9 | $\fcolorbox{gray}{#90fda9}{\phantom{x}}$ |
|         xkcd:grey/green         | #86a17d | $\fcolorbox{gray}{#86a17d}{\phantom{x}}$ |
|         xkcd:light gold         | #fddc5c | $\fcolorbox{gray}{#fddc5c}{\phantom{x}}$ |
|        xkcd:seafoam blue        | #78d1b6 | $\fcolorbox{gray}{#78d1b6}{\phantom{x}}$ |
|           xkcd:topaz            | #13bbaf | $\fcolorbox{gray}{#13bbaf}{\phantom{x}}$ |
|        xkcd:violet pink         | #fb5ffc | $\fcolorbox{gray}{#fb5ffc}{\phantom{x}}$ |
|        xkcd:wintergreen         | #20f986 | $\fcolorbox{gray}{#20f986}{\phantom{x}}$ |
|         xkcd:yellow tan         | #ffe36e | $\fcolorbox{gray}{#ffe36e}{\phantom{x}}$ |
|        xkcd:dark fuchsia        | #9d0759 | $\fcolorbox{gray}{#9d0759}{\phantom{x}}$ |
|        xkcd:indigo blue         | #3a18b1 | $\fcolorbox{gray}{#3a18b1}{\phantom{x}}$ |
|   xkcd:light yellowish green    | #c2ff89 | $\fcolorbox{gray}{#c2ff89}{\phantom{x}}$ |
|        xkcd:pale magenta        | #d767ad | $\fcolorbox{gray}{#d767ad}{\phantom{x}}$ |
|        xkcd:rich purple         | #720058 | $\fcolorbox{gray}{#720058}{\phantom{x}}$ |
|      xkcd:sunflower yellow      | #ffda03 | $\fcolorbox{gray}{#ffda03}{\phantom{x}}$ |
|         xkcd:green/blue         | #01c08d | $\fcolorbox{gray}{#01c08d}{\phantom{x}}$ |
|          xkcd:leather           | #ac7434 | $\fcolorbox{gray}{#ac7434}{\phantom{x}}$ |
|        xkcd:racing green        | #014600 | $\fcolorbox{gray}{#014600}{\phantom{x}}$ |
|        xkcd:vivid purple        | #9900fa | $\fcolorbox{gray}{#9900fa}{\phantom{x}}$ |
|      xkcd:dark royal blue       | #02066f | $\fcolorbox{gray}{#02066f}{\phantom{x}}$ |
|           xkcd:hazel            | #8e7618 | $\fcolorbox{gray}{#8e7618}{\phantom{x}}$ |
|         xkcd:muted pink         | #d1768f | $\fcolorbox{gray}{#d1768f}{\phantom{x}}$ |
|        xkcd:booger green        | #96b403 | $\fcolorbox{gray}{#96b403}{\phantom{x}}$ |
|           xkcd:canary           | #fdff63 | $\fcolorbox{gray}{#fdff63}{\phantom{x}}$ |
|         xkcd:cool grey          | #95a3a6 | $\fcolorbox{gray}{#95a3a6}{\phantom{x}}$ |
|         xkcd:dark taupe         | #7f684e | $\fcolorbox{gray}{#7f684e}{\phantom{x}}$ |
|       xkcd:darkish purple       | #751973 | $\fcolorbox{gray}{#751973}{\phantom{x}}$ |
|         xkcd:true green         | #089404 | $\fcolorbox{gray}{#089404}{\phantom{x}}$ |
|         xkcd:coral pink         | #ff6163 | $\fcolorbox{gray}{#ff6163}{\phantom{x}}$ |
|         xkcd:dark sage          | #598556 | $\fcolorbox{gray}{#598556}{\phantom{x}}$ |
|      xkcd:dark slate blue       | #214761 | $\fcolorbox{gray}{#214761}{\phantom{x}}$ |
|         xkcd:flat blue          | #3c73a8 | $\fcolorbox{gray}{#3c73a8}{\phantom{x}}$ |
|          xkcd:mushroom          | #ba9e88 | $\fcolorbox{gray}{#ba9e88}{\phantom{x}}$ |
|         xkcd:rich blue          | #021bf9 | $\fcolorbox{gray}{#021bf9}{\phantom{x}}$ |
|        xkcd:dirty purple        | #734a65 | $\fcolorbox{gray}{#734a65}{\phantom{x}}$ |
|         xkcd:greenblue          | #23c48b | $\fcolorbox{gray}{#23c48b}{\phantom{x}}$ |
|         xkcd:icky green         | #8fae22 | $\fcolorbox{gray}{#8fae22}{\phantom{x}}$ |
|        xkcd:light khaki         | #e6f2a2 | $\fcolorbox{gray}{#e6f2a2}{\phantom{x}}$ |
|         xkcd:warm blue          | #4b57db | $\fcolorbox{gray}{#4b57db}{\phantom{x}}$ |
|       xkcd:dark hot pink        | #d90166 | $\fcolorbox{gray}{#d90166}{\phantom{x}}$ |
|       xkcd:deep sea blue        | #015482 | $\fcolorbox{gray}{#015482}{\phantom{x}}$ |
|          xkcd:carmine           | #9d0216 | $\fcolorbox{gray}{#9d0216}{\phantom{x}}$ |
|     xkcd:dark yellow green      | #728f02 | $\fcolorbox{gray}{#728f02}{\phantom{x}}$ |
|         xkcd:pale peach         | #ffe5ad | $\fcolorbox{gray}{#ffe5ad}{\phantom{x}}$ |
|        xkcd:plum purple         | #4e0550 | $\fcolorbox{gray}{#4e0550}{\phantom{x}}$ |
|         xkcd:golden rod         | #f9bc08 | $\fcolorbox{gray}{#f9bc08}{\phantom{x}}$ |
|          xkcd:neon red          | #ff073a | $\fcolorbox{gray}{#ff073a}{\phantom{x}}$ |
|          xkcd:old pink          | #c77986 | $\fcolorbox{gray}{#c77986}{\phantom{x}}$ |
|       xkcd:very pale blue       | #d6fffe | $\fcolorbox{gray}{#d6fffe}{\phantom{x}}$ |
|        xkcd:blood orange        | #fe4b03 | $\fcolorbox{gray}{#fe4b03}{\phantom{x}}$ |
|         xkcd:grapefruit         | #fd5956 | $\fcolorbox{gray}{#fd5956}{\phantom{x}}$ |
|        xkcd:sand yellow         | #fce166 | $\fcolorbox{gray}{#fce166}{\phantom{x}}$ |
|         xkcd:clay brown         | #b2713d | $\fcolorbox{gray}{#b2713d}{\phantom{x}}$ |
|       xkcd:dark blue grey       | #1f3b4d | $\fcolorbox{gray}{#1f3b4d}{\phantom{x}}$ |
|         xkcd:flat green         | #699d4c | $\fcolorbox{gray}{#699d4c}{\phantom{x}}$ |
|      xkcd:light green blue      | #56fca2 | $\fcolorbox{gray}{#56fca2}{\phantom{x}}$ |
|         xkcd:warm pink          | #fb5581 | $\fcolorbox{gray}{#fb5581}{\phantom{x}}$ |
|        xkcd:dodger blue         | #3e82fc | $\fcolorbox{gray}{#3e82fc}{\phantom{x}}$ |
|        xkcd:gross green         | #a0bf16 | $\fcolorbox{gray}{#a0bf16}{\phantom{x}}$ |
|            xkcd:ice             | #d6fffa | $\fcolorbox{gray}{#d6fffa}{\phantom{x}}$ |
|       xkcd:metallic blue        | #4f738e | $\fcolorbox{gray}{#4f738e}{\phantom{x}}$ |
|        xkcd:pale salmon         | #ffb19a | $\fcolorbox{gray}{#ffb19a}{\phantom{x}}$ |
|         xkcd:sap green          | #5c8b15 | $\fcolorbox{gray}{#5c8b15}{\phantom{x}}$ |
|           xkcd:algae            | #54ac68 | $\fcolorbox{gray}{#54ac68}{\phantom{x}}$ |
|         xkcd:bluey grey         | #89a0b0 | $\fcolorbox{gray}{#89a0b0}{\phantom{x}}$ |
|        xkcd:greeny grey         | #7ea07a | $\fcolorbox{gray}{#7ea07a}{\phantom{x}}$ |
|     xkcd:highlighter green      | #1bfc06 | $\fcolorbox{gray}{#1bfc06}{\phantom{x}}$ |
|      xkcd:light light blue      | #cafffb | $\fcolorbox{gray}{#cafffb}{\phantom{x}}$ |
|         xkcd:light mint         | #b6ffbb | $\fcolorbox{gray}{#b6ffbb}{\phantom{x}}$ |
|         xkcd:raw umber          | #a75e09 | $\fcolorbox{gray}{#a75e09}{\phantom{x}}$ |
|         xkcd:vivid blue         | #152eff | $\fcolorbox{gray}{#152eff}{\phantom{x}}$ |
|       xkcd:deep lavender        | #8d5eb7 | $\fcolorbox{gray}{#8d5eb7}{\phantom{x}}$ |
|         xkcd:dull teal          | #5f9e8f | $\fcolorbox{gray}{#5f9e8f}{\phantom{x}}$ |
|    xkcd:light greenish blue     | #63f7b4 | $\fcolorbox{gray}{#63f7b4}{\phantom{x}}$ |
|         xkcd:mud green          | #606602 | $\fcolorbox{gray}{#606602}{\phantom{x}}$ |
|           xkcd:pinky            | #fc86aa | $\fcolorbox{gray}{#fc86aa}{\phantom{x}}$ |
|          xkcd:red wine          | #8c0034 | $\fcolorbox{gray}{#8c0034}{\phantom{x}}$ |
|         xkcd:shit green         | #758000 | $\fcolorbox{gray}{#758000}{\phantom{x}}$ |
|         xkcd:tan brown          | #ab7e4c | $\fcolorbox{gray}{#ab7e4c}{\phantom{x}}$ |
|          xkcd:darkblue          | #030764 | $\fcolorbox{gray}{#030764}{\phantom{x}}$ |
|            xkcd:rosa            | #fe86a4 | $\fcolorbox{gray}{#fe86a4}{\phantom{x}}$ |
|          xkcd:lipstick          | #d5174e | $\fcolorbox{gray}{#d5174e}{\phantom{x}}$ |
|         xkcd:pale mauve         | #fed0fc | $\fcolorbox{gray}{#fed0fc}{\phantom{x}}$ |
|           xkcd:claret           | #680018 | $\fcolorbox{gray}{#680018}{\phantom{x}}$ |
|         xkcd:dandelion          | #fedf08 | $\fcolorbox{gray}{#fedf08}{\phantom{x}}$ |
|         xkcd:orangered          | #fe420f | $\fcolorbox{gray}{#fe420f}{\phantom{x}}$ |
|         xkcd:poop green         | #6f7c00 | $\fcolorbox{gray}{#6f7c00}{\phantom{x}}$ |
|            xkcd:ruby            | #ca0147 | $\fcolorbox{gray}{#ca0147}{\phantom{x}}$ |
|            xkcd:dark            | #1b2431 | $\fcolorbox{gray}{#1b2431}{\phantom{x}}$ |
|     xkcd:greenish turquoise     | #00fbb0 | $\fcolorbox{gray}{#00fbb0}{\phantom{x}}$ |
|         xkcd:pastel red         | #db5856 | $\fcolorbox{gray}{#db5856}{\phantom{x}}$ |
|        xkcd:piss yellow         | #ddd618 | $\fcolorbox{gray}{#ddd618}{\phantom{x}}$ |
|        xkcd:bright cyan         | #41fdfe | $\fcolorbox{gray}{#41fdfe}{\phantom{x}}$ |
|         xkcd:dark coral         | #cf524e | $\fcolorbox{gray}{#cf524e}{\phantom{x}}$ |
|        xkcd:algae green         | #21c36f | $\fcolorbox{gray}{#21c36f}{\phantom{x}}$ |
|        xkcd:darkish red         | #a90308 | $\fcolorbox{gray}{#a90308}{\phantom{x}}$ |
|        xkcd:reddy brown         | #6e1005 | $\fcolorbox{gray}{#6e1005}{\phantom{x}}$ |
|         xkcd:blush pink         | #fe828c | $\fcolorbox{gray}{#fe828c}{\phantom{x}}$ |
|      xkcd:camouflage green      | #4b6113 | $\fcolorbox{gray}{#4b6113}{\phantom{x}}$ |
|         xkcd:lawn green         | #4da409 | $\fcolorbox{gray}{#4da409}{\phantom{x}}$ |
|           xkcd:putty            | #beae8a | $\fcolorbox{gray}{#beae8a}{\phantom{x}}$ |
|        xkcd:vibrant blue        | #0339f8 | $\fcolorbox{gray}{#0339f8}{\phantom{x}}$ |
|         xkcd:dark sand          | #a88f59 | $\fcolorbox{gray}{#a88f59}{\phantom{x}}$ |
|        xkcd:purple/blue         | #5d21d0 | $\fcolorbox{gray}{#5d21d0}{\phantom{x}}$ |
|          xkcd:saffron           | #feb209 | $\fcolorbox{gray}{#feb209}{\phantom{x}}$ |
|          xkcd:twilight          | #4e518b | $\fcolorbox{gray}{#4e518b}{\phantom{x}}$ |
|         xkcd:warm brown         | #964e02 | $\fcolorbox{gray}{#964e02}{\phantom{x}}$ |
|          xkcd:bluegrey          | #85a3b2 | $\fcolorbox{gray}{#85a3b2}{\phantom{x}}$ |
|      xkcd:bubble gum pink       | #ff69af | $\fcolorbox{gray}{#ff69af}{\phantom{x}}$ |
|       xkcd:duck egg blue        | #c3fbf4 | $\fcolorbox{gray}{#c3fbf4}{\phantom{x}}$ |
|       xkcd:greenish cyan        | #2afeb7 | $\fcolorbox{gray}{#2afeb7}{\phantom{x}}$ |
|           xkcd:petrol           | #005f6a | $\fcolorbox{gray}{#005f6a}{\phantom{x}}$ |
|           xkcd:royal            | #0c1793 | $\fcolorbox{gray}{#0c1793}{\phantom{x}}$ |
|           xkcd:butter           | #ffff81 | $\fcolorbox{gray}{#ffff81}{\phantom{x}}$ |
|        xkcd:dusty orange        | #f0833a | $\fcolorbox{gray}{#f0833a}{\phantom{x}}$ |
|         xkcd:off yellow         | #f1f33f | $\fcolorbox{gray}{#f1f33f}{\phantom{x}}$ |
|      xkcd:pale olive green      | #b1d27b | $\fcolorbox{gray}{#b1d27b}{\phantom{x}}$ |
|          xkcd:orangish          | #fc824a | $\fcolorbox{gray}{#fc824a}{\phantom{x}}$ |
|            xkcd:leaf            | #71aa34 | $\fcolorbox{gray}{#71aa34}{\phantom{x}}$ |
|      xkcd:light blue grey       | #b7c9e2 | $\fcolorbox{gray}{#b7c9e2}{\phantom{x}}$ |
|        xkcd:dried blood         | #4b0101 | $\fcolorbox{gray}{#4b0101}{\phantom{x}}$ |
|      xkcd:lightish purple       | #a552e6 | $\fcolorbox{gray}{#a552e6}{\phantom{x}}$ |
|         xkcd:rusty red          | #af2f0d | $\fcolorbox{gray}{#af2f0d}{\phantom{x}}$ |
|       xkcd:lavender blue        | #8b88f8 | $\fcolorbox{gray}{#8b88f8}{\phantom{x}}$ |
|     xkcd:light grass green      | #9af764 | $\fcolorbox{gray}{#9af764}{\phantom{x}}$ |
|      xkcd:light mint green      | #a6fbb2 | $\fcolorbox{gray}{#a6fbb2}{\phantom{x}}$ |
|         xkcd:sunflower          | #ffc512 | $\fcolorbox{gray}{#ffc512}{\phantom{x}}$ |
|           xkcd:velvet           | #750851 | $\fcolorbox{gray}{#750851}{\phantom{x}}$ |
|        xkcd:brick orange        | #c14a09 | $\fcolorbox{gray}{#c14a09}{\phantom{x}}$ |
|        xkcd:lightish red        | #fe2f4a | $\fcolorbox{gray}{#fe2f4a}{\phantom{x}}$ |
|         xkcd:pure blue          | #0203e2 | $\fcolorbox{gray}{#0203e2}{\phantom{x}}$ |
|       xkcd:twilight blue        | #0a437a | $\fcolorbox{gray}{#0a437a}{\phantom{x}}$ |
|         xkcd:violet red         | #a50055 | $\fcolorbox{gray}{#a50055}{\phantom{x}}$ |
|       xkcd:yellowy brown        | #ae8b0c | $\fcolorbox{gray}{#ae8b0c}{\phantom{x}}$ |
|         xkcd:carnation          | #fd798f | $\fcolorbox{gray}{#fd798f}{\phantom{x}}$ |
|        xkcd:muddy yellow        | #bfac05 | $\fcolorbox{gray}{#bfac05}{\phantom{x}}$ |
|     xkcd:dark seafoam green     | #3eaf76 | $\fcolorbox{gray}{#3eaf76}{\phantom{x}}$ |
|         xkcd:deep rose          | #c74767 | $\fcolorbox{gray}{#c74767}{\phantom{x}}$ |
|         xkcd:dusty red          | #b9484e | $\fcolorbox{gray}{#b9484e}{\phantom{x}}$ |
|         xkcd:grey/blue          | #647d8e | $\fcolorbox{gray}{#647d8e}{\phantom{x}}$ |
|         xkcd:lemon lime         | #bffe28 | $\fcolorbox{gray}{#bffe28}{\phantom{x}}$ |
|        xkcd:purple/pink         | #d725de | $\fcolorbox{gray}{#d725de}{\phantom{x}}$ |
|        xkcd:brown yellow        | #b29705 | $\fcolorbox{gray}{#b29705}{\phantom{x}}$ |
|        xkcd:purple brown        | #673a3f | $\fcolorbox{gray}{#673a3f}{\phantom{x}}$ |
|          xkcd:wisteria          | #a87dc2 | $\fcolorbox{gray}{#a87dc2}{\phantom{x}}$ |
|       xkcd:banana yellow        | #fafe4b | $\fcolorbox{gray}{#fafe4b}{\phantom{x}}$ |
|        xkcd:lipstick red        | #c0022f | $\fcolorbox{gray}{#c0022f}{\phantom{x}}$ |
|         xkcd:water blue         | #0e87cc | $\fcolorbox{gray}{#0e87cc}{\phantom{x}}$ |
|         xkcd:brown grey         | #8d8468 | $\fcolorbox{gray}{#8d8468}{\phantom{x}}$ |
|       xkcd:vibrant purple       | #ad03de | $\fcolorbox{gray}{#ad03de}{\phantom{x}}$ |
|         xkcd:baby green         | #8cff9e | $\fcolorbox{gray}{#8cff9e}{\phantom{x}}$ |
|         xkcd:barf green         | #94ac02 | $\fcolorbox{gray}{#94ac02}{\phantom{x}}$ |
|       xkcd:eggshell blue        | #c4fff7 | $\fcolorbox{gray}{#c4fff7}{\phantom{x}}$ |
|        xkcd:sandy yellow        | #fdee73 | $\fcolorbox{gray}{#fdee73}{\phantom{x}}$ |
|         xkcd:cool green         | #33b864 | $\fcolorbox{gray}{#33b864}{\phantom{x}}$ |
|            xkcd:pale            | #fff9d0 | $\fcolorbox{gray}{#fff9d0}{\phantom{x}}$ |
|         xkcd:blue/grey          | #758da3 | $\fcolorbox{gray}{#758da3}{\phantom{x}}$ |
|        xkcd:hot magenta         | #f504c9 | $\fcolorbox{gray}{#f504c9}{\phantom{x}}$ |
|          xkcd:greyblue          | #77a1b5 | $\fcolorbox{gray}{#77a1b5}{\phantom{x}}$ |
|          xkcd:purpley           | #8756e4 | $\fcolorbox{gray}{#8756e4}{\phantom{x}}$ |
|      xkcd:baby shit green       | #889717 | $\fcolorbox{gray}{#889717}{\phantom{x}}$ |
|       xkcd:brownish pink        | #c27e79 | $\fcolorbox{gray}{#c27e79}{\phantom{x}}$ |
|      xkcd:dark aquamarine       | #017371 | $\fcolorbox{gray}{#017371}{\phantom{x}}$ |
|          xkcd:diarrhea          | #9f8303 | $\fcolorbox{gray}{#9f8303}{\phantom{x}}$ |
|       xkcd:light mustard        | #f7d560 | $\fcolorbox{gray}{#f7d560}{\phantom{x}}$ |
|       xkcd:pale sky blue        | #bdf6fe | $\fcolorbox{gray}{#bdf6fe}{\phantom{x}}$ |
|        xkcd:turtle green        | #75b84f | $\fcolorbox{gray}{#75b84f}{\phantom{x}}$ |
|        xkcd:bright olive        | #9cbb04 | $\fcolorbox{gray}{#9cbb04}{\phantom{x}}$ |
|       xkcd:dark grey blue       | #29465b | $\fcolorbox{gray}{#29465b}{\phantom{x}}$ |
|        xkcd:greeny brown        | #696006 | $\fcolorbox{gray}{#696006}{\phantom{x}}$ |
|        xkcd:lemon green         | #adf802 | $\fcolorbox{gray}{#adf802}{\phantom{x}}$ |
|      xkcd:light periwinkle      | #c1c6fc | $\fcolorbox{gray}{#c1c6fc}{\phantom{x}}$ |
|       xkcd:seaweed green        | #35ad6b | $\fcolorbox{gray}{#35ad6b}{\phantom{x}}$ |
|      xkcd:sunshine yellow       | #fffd37 | $\fcolorbox{gray}{#fffd37}{\phantom{x}}$ |
|        xkcd:ugly purple         | #a442a0 | $\fcolorbox{gray}{#a442a0}{\phantom{x}}$ |
|        xkcd:medium pink         | #f36196 | $\fcolorbox{gray}{#f36196}{\phantom{x}}$ |
|         xkcd:puke brown         | #947706 | $\fcolorbox{gray}{#947706}{\phantom{x}}$ |
|      xkcd:very light pink       | #fff4f2 | $\fcolorbox{gray}{#fff4f2}{\phantom{x}}$ |
|          xkcd:viridian          | #1e9167 | $\fcolorbox{gray}{#1e9167}{\phantom{x}}$ |
|            xkcd:bile            | #b5c306 | $\fcolorbox{gray}{#b5c306}{\phantom{x}}$ |
|        xkcd:faded yellow        | #feff7f | $\fcolorbox{gray}{#feff7f}{\phantom{x}}$ |
|      xkcd:very pale green       | #cffdbc | $\fcolorbox{gray}{#cffdbc}{\phantom{x}}$ |
|       xkcd:vibrant green        | #0add08 | $\fcolorbox{gray}{#0add08}{\phantom{x}}$ |
|        xkcd:bright lime         | #87fd05 | $\fcolorbox{gray}{#87fd05}{\phantom{x}}$ |
|         xkcd:spearmint          | #1ef876 | $\fcolorbox{gray}{#1ef876}{\phantom{x}}$ |
|      xkcd:light aquamarine      | #7bfdc7 | $\fcolorbox{gray}{#7bfdc7}{\phantom{x}}$ |
|         xkcd:light sage         | #bcecac | $\fcolorbox{gray}{#bcecac}{\phantom{x}}$ |
|        xkcd:yellowgreen         | #bbf90f | $\fcolorbox{gray}{#bbf90f}{\phantom{x}}$ |
|          xkcd:baby poo          | #ab9004 | $\fcolorbox{gray}{#ab9004}{\phantom{x}}$ |
|        xkcd:dark seafoam        | #1fb57a | $\fcolorbox{gray}{#1fb57a}{\phantom{x}}$ |
|         xkcd:deep teal          | #00555a | $\fcolorbox{gray}{#00555a}{\phantom{x}}$ |
|          xkcd:heather           | #a484ac | $\fcolorbox{gray}{#a484ac}{\phantom{x}}$ |
|        xkcd:rust orange         | #c45508 | $\fcolorbox{gray}{#c45508}{\phantom{x}}$ |
|         xkcd:dirty blue         | #3f829d | $\fcolorbox{gray}{#3f829d}{\phantom{x}}$ |
|         xkcd:fern green         | #548d44 | $\fcolorbox{gray}{#548d44}{\phantom{x}}$ |
|        xkcd:bright lilac        | #c95efb | $\fcolorbox{gray}{#c95efb}{\phantom{x}}$ |
|        xkcd:weird green         | #3ae57f | $\fcolorbox{gray}{#3ae57f}{\phantom{x}}$ |
|        xkcd:peacock blue        | #016795 | $\fcolorbox{gray}{#016795}{\phantom{x}}$ |
|       xkcd:avocado green        | #87a922 | $\fcolorbox{gray}{#87a922}{\phantom{x}}$ |
|        xkcd:faded orange        | #f0944d | $\fcolorbox{gray}{#f0944d}{\phantom{x}}$ |
|        xkcd:grape purple        | #5d1451 | $\fcolorbox{gray}{#5d1451}{\phantom{x}}$ |
|         xkcd:hot green          | #25ff29 | $\fcolorbox{gray}{#25ff29}{\phantom{x}}$ |
|        xkcd:lime yellow         | #d0fe1d | $\fcolorbox{gray}{#d0fe1d}{\phantom{x}}$ |
|           xkcd:mango            | #ffa62b | $\fcolorbox{gray}{#ffa62b}{\phantom{x}}$ |
|          xkcd:shamrock          | #01b44c | $\fcolorbox{gray}{#01b44c}{\phantom{x}}$ |
|         xkcd:bubblegum          | #ff6cb5 | $\fcolorbox{gray}{#ff6cb5}{\phantom{x}}$ |
|       xkcd:purplish brown       | #6b4247 | $\fcolorbox{gray}{#6b4247}{\phantom{x}}$ |
|        xkcd:vomit yellow        | #c7c10c | $\fcolorbox{gray}{#c7c10c}{\phantom{x}}$ |
|         xkcd:pale cyan          | #b7fffa | $\fcolorbox{gray}{#b7fffa}{\phantom{x}}$ |
|          xkcd:key lime          | #aeff6e | $\fcolorbox{gray}{#aeff6e}{\phantom{x}}$ |
|         xkcd:tomato red         | #ec2d01 | $\fcolorbox{gray}{#ec2d01}{\phantom{x}}$ |
|         xkcd:lightgreen         | #76ff7b | $\fcolorbox{gray}{#76ff7b}{\phantom{x}}$ |
|           xkcd:merlot           | #730039 | $\fcolorbox{gray}{#730039}{\phantom{x}}$ |
|         xkcd:night blue         | #040348 | $\fcolorbox{gray}{#040348}{\phantom{x}}$ |
|       xkcd:purpleish pink       | #df4ec8 | $\fcolorbox{gray}{#df4ec8}{\phantom{x}}$ |
|           xkcd:apple            | #6ecb3c | $\fcolorbox{gray}{#6ecb3c}{\phantom{x}}$ |
|      xkcd:baby poop green       | #8f9805 | $\fcolorbox{gray}{#8f9805}{\phantom{x}}$ |
|        xkcd:green apple         | #5edc1f | $\fcolorbox{gray}{#5edc1f}{\phantom{x}}$ |
|         xkcd:heliotrope         | #d94ff5 | $\fcolorbox{gray}{#d94ff5}{\phantom{x}}$ |
|        xkcd:yellow/green        | #c8fd3d | $\fcolorbox{gray}{#c8fd3d}{\phantom{x}}$ |
|        xkcd:almost black        | #070d0d | $\fcolorbox{gray}{#070d0d}{\phantom{x}}$ |
|         xkcd:cool blue          | #4984b8 | $\fcolorbox{gray}{#4984b8}{\phantom{x}}$ |
|        xkcd:leafy green         | #51b73b | $\fcolorbox{gray}{#51b73b}{\phantom{x}}$ |
|       xkcd:mustard brown        | #ac7e04 | $\fcolorbox{gray}{#ac7e04}{\phantom{x}}$ |
|            xkcd:dusk            | #4e5481 | $\fcolorbox{gray}{#4e5481}{\phantom{x}}$ |
|         xkcd:dull brown         | #876e4b | $\fcolorbox{gray}{#876e4b}{\phantom{x}}$ |
|         xkcd:frog green         | #58bc08 | $\fcolorbox{gray}{#58bc08}{\phantom{x}}$ |
|        xkcd:vivid green         | #2fef10 | $\fcolorbox{gray}{#2fef10}{\phantom{x}}$ |
|     xkcd:bright light green     | #2dfe54 | $\fcolorbox{gray}{#2dfe54}{\phantom{x}}$ |
|        xkcd:fluro green         | #0aff02 | $\fcolorbox{gray}{#0aff02}{\phantom{x}}$ |
|            xkcd:kiwi            | #9cef43 | $\fcolorbox{gray}{#9cef43}{\phantom{x}}$ |
|          xkcd:seaweed           | #18d17b | $\fcolorbox{gray}{#18d17b}{\phantom{x}}$ |
|         xkcd:navy green         | #35530a | $\fcolorbox{gray}{#35530a}{\phantom{x}}$ |
|      xkcd:ultramarine blue      | #1805db | $\fcolorbox{gray}{#1805db}{\phantom{x}}$ |
|            xkcd:iris            | #6258c4 | $\fcolorbox{gray}{#6258c4}{\phantom{x}}$ |
|       xkcd:pastel orange        | #ff964f | $\fcolorbox{gray}{#ff964f}{\phantom{x}}$ |
|      xkcd:yellowish orange      | #ffab0f | $\fcolorbox{gray}{#ffab0f}{\phantom{x}}$ |
|        xkcd:perrywinkle         | #8f8ce7 | $\fcolorbox{gray}{#8f8ce7}{\phantom{x}}$ |
|          xkcd:tealish           | #24bca8 | $\fcolorbox{gray}{#24bca8}{\phantom{x}}$ |
|         xkcd:dark plum          | #3f012c | $\fcolorbox{gray}{#3f012c}{\phantom{x}}$ |
|            xkcd:pear            | #cbf85f | $\fcolorbox{gray}{#cbf85f}{\phantom{x}}$ |
|       xkcd:pinkish orange       | #ff724c | $\fcolorbox{gray}{#ff724c}{\phantom{x}}$ |
|      xkcd:midnight purple       | #280137 | $\fcolorbox{gray}{#280137}{\phantom{x}}$ |
|        xkcd:light urple         | #b36ff6 | $\fcolorbox{gray}{#b36ff6}{\phantom{x}}$ |
|         xkcd:dark mint          | #48c072 | $\fcolorbox{gray}{#48c072}{\phantom{x}}$ |
|        xkcd:greenish tan        | #bccb7a | $\fcolorbox{gray}{#bccb7a}{\phantom{x}}$ |
|       xkcd:light burgundy       | #a8415b | $\fcolorbox{gray}{#a8415b}{\phantom{x}}$ |
|       xkcd:turquoise blue       | #06b1c4 | $\fcolorbox{gray}{#06b1c4}{\phantom{x}}$ |
|         xkcd:ugly pink          | #cd7584 | $\fcolorbox{gray}{#cd7584}{\phantom{x}}$ |
|           xkcd:sandy            | #f1da7a | $\fcolorbox{gray}{#f1da7a}{\phantom{x}}$ |
|       xkcd:electric pink        | #ff0490 | $\fcolorbox{gray}{#ff0490}{\phantom{x}}$ |
|        xkcd:muted purple        | #805b87 | $\fcolorbox{gray}{#805b87}{\phantom{x}}$ |
|         xkcd:mid green          | #50a747 | $\fcolorbox{gray}{#50a747}{\phantom{x}}$ |
|          xkcd:greyish           | #a8a495 | $\fcolorbox{gray}{#a8a495}{\phantom{x}}$ |
|        xkcd:neon yellow         | #cfff04 | $\fcolorbox{gray}{#cfff04}{\phantom{x}}$ |
|           xkcd:banana           | #ffff7e | $\fcolorbox{gray}{#ffff7e}{\phantom{x}}$ |
|       xkcd:carnation pink       | #ff7fa7 | $\fcolorbox{gray}{#ff7fa7}{\phantom{x}}$ |
|           xkcd:tomato           | #ef4026 | $\fcolorbox{gray}{#ef4026}{\phantom{x}}$ |
|            xkcd:sea             | #3c9992 | $\fcolorbox{gray}{#3c9992}{\phantom{x}}$ |
|        xkcd:muddy brown         | #886806 | $\fcolorbox{gray}{#886806}{\phantom{x}}$ |
|      xkcd:turquoise green       | #04f489 | $\fcolorbox{gray}{#04f489}{\phantom{x}}$ |
|            xkcd:buff            | #fef69e | $\fcolorbox{gray}{#fef69e}{\phantom{x}}$ |
|            xkcd:fawn            | #cfaf7b | $\fcolorbox{gray}{#cfaf7b}{\phantom{x}}$ |
|         xkcd:muted blue         | #3b719f | $\fcolorbox{gray}{#3b719f}{\phantom{x}}$ |
|         xkcd:pale rose          | #fdc1c5 | $\fcolorbox{gray}{#fdc1c5}{\phantom{x}}$ |
|      xkcd:dark mint green       | #20c073 | $\fcolorbox{gray}{#20c073}{\phantom{x}}$ |
|          xkcd:amethyst          | #9b5fc0 | $\fcolorbox{gray}{#9b5fc0}{\phantom{x}}$ |
|         xkcd:blue/green         | #0f9b8e | $\fcolorbox{gray}{#0f9b8e}{\phantom{x}}$ |
|          xkcd:chestnut          | #742802 | $\fcolorbox{gray}{#742802}{\phantom{x}}$ |
|         xkcd:sick green         | #9db92c | $\fcolorbox{gray}{#9db92c}{\phantom{x}}$ |
|            xkcd:pea             | #a4bf20 | $\fcolorbox{gray}{#a4bf20}{\phantom{x}}$ |
|        xkcd:rusty orange        | #cd5909 | $\fcolorbox{gray}{#cd5909}{\phantom{x}}$ |
|           xkcd:stone            | #ada587 | $\fcolorbox{gray}{#ada587}{\phantom{x}}$ |
|          xkcd:rose red          | #be013c | $\fcolorbox{gray}{#be013c}{\phantom{x}}$ |
|         xkcd:pale aqua          | #b8ffeb | $\fcolorbox{gray}{#b8ffeb}{\phantom{x}}$ |
|        xkcd:deep orange         | #dc4d01 | $\fcolorbox{gray}{#dc4d01}{\phantom{x}}$ |
|           xkcd:earth            | #a2653e | $\fcolorbox{gray}{#a2653e}{\phantom{x}}$ |
|        xkcd:mossy green         | #638b27 | $\fcolorbox{gray}{#638b27}{\phantom{x}}$ |
|        xkcd:grassy green        | #419c03 | $\fcolorbox{gray}{#419c03}{\phantom{x}}$ |
|      xkcd:pale lime green       | #b1ff65 | $\fcolorbox{gray}{#b1ff65}{\phantom{x}}$ |
|      xkcd:light grey blue       | #9dbcd4 | $\fcolorbox{gray}{#9dbcd4}{\phantom{x}}$ |
|         xkcd:pale grey          | #fdfdfe | $\fcolorbox{gray}{#fdfdfe}{\phantom{x}}$ |
|         xkcd:asparagus          | #77ab56 | $\fcolorbox{gray}{#77ab56}{\phantom{x}}$ |
|         xkcd:blueberry          | #464196 | $\fcolorbox{gray}{#464196}{\phantom{x}}$ |
|         xkcd:purple red         | #990147 | $\fcolorbox{gray}{#990147}{\phantom{x}}$ |
|         xkcd:pale lime          | #befd73 | $\fcolorbox{gray}{#befd73}{\phantom{x}}$ |
|       xkcd:greenish teal        | #32bf84 | $\fcolorbox{gray}{#32bf84}{\phantom{x}}$ |
|          xkcd:caramel           | #af6f09 | $\fcolorbox{gray}{#af6f09}{\phantom{x}}$ |
|        xkcd:deep magenta        | #a0025c | $\fcolorbox{gray}{#a0025c}{\phantom{x}}$ |
|        xkcd:light peach         | #ffd8b1 | $\fcolorbox{gray}{#ffd8b1}{\phantom{x}}$ |
|       xkcd:milk chocolate       | #7f4e1e | $\fcolorbox{gray}{#7f4e1e}{\phantom{x}}$ |
|           xkcd:ocher            | #bf9b0c | $\fcolorbox{gray}{#bf9b0c}{\phantom{x}}$ |
|         xkcd:off green          | #6ba353 | $\fcolorbox{gray}{#6ba353}{\phantom{x}}$ |
|        xkcd:purply pink         | #f075e6 | $\fcolorbox{gray}{#f075e6}{\phantom{x}}$ |
|         xkcd:lightblue          | #7bc8f6 | $\fcolorbox{gray}{#7bc8f6}{\phantom{x}}$ |
|         xkcd:dusky blue         | #475f94 | $\fcolorbox{gray}{#475f94}{\phantom{x}}$ |
|           xkcd:golden           | #f5bf03 | $\fcolorbox{gray}{#f5bf03}{\phantom{x}}$ |
|        xkcd:light beige         | #fffeb6 | $\fcolorbox{gray}{#fffeb6}{\phantom{x}}$ |
|       xkcd:butter yellow        | #fffd74 | $\fcolorbox{gray}{#fffd74}{\phantom{x}}$ |
|        xkcd:dusky purple        | #895b7b | $\fcolorbox{gray}{#895b7b}{\phantom{x}}$ |
|        xkcd:french blue         | #436bad | $\fcolorbox{gray}{#436bad}{\phantom{x}}$ |
|        xkcd:ugly yellow         | #d0c101 | $\fcolorbox{gray}{#d0c101}{\phantom{x}}$ |
|       xkcd:greeny yellow        | #c6f808 | $\fcolorbox{gray}{#c6f808}{\phantom{x}}$ |
|        xkcd:orangish red        | #f43605 | $\fcolorbox{gray}{#f43605}{\phantom{x}}$ |
|       xkcd:shamrock green       | #02c14d | $\fcolorbox{gray}{#02c14d}{\phantom{x}}$ |
|       xkcd:orangish brown       | #b25f03 | $\fcolorbox{gray}{#b25f03}{\phantom{x}}$ |
|         xkcd:tree green         | #2a7e19 | $\fcolorbox{gray}{#2a7e19}{\phantom{x}}$ |
|        xkcd:deep violet         | #490648 | $\fcolorbox{gray}{#490648}{\phantom{x}}$ |
|          xkcd:gunmetal          | #536267 | $\fcolorbox{gray}{#536267}{\phantom{x}}$ |
|        xkcd:blue/purple         | #5a06ef | $\fcolorbox{gray}{#5a06ef}{\phantom{x}}$ |
|           xkcd:cherry           | #cf0234 | $\fcolorbox{gray}{#cf0234}{\phantom{x}}$ |
|        xkcd:sandy brown         | #c4a661 | $\fcolorbox{gray}{#c4a661}{\phantom{x}}$ |
|         xkcd:warm grey          | #978a84 | $\fcolorbox{gray}{#978a84}{\phantom{x}}$ |
|        xkcd:dark indigo         | #1f0954 | $\fcolorbox{gray}{#1f0954}{\phantom{x}}$ |
|          xkcd:midnight          | #03012d | $\fcolorbox{gray}{#03012d}{\phantom{x}}$ |
|        xkcd:bluey green         | #2bb179 | $\fcolorbox{gray}{#2bb179}{\phantom{x}}$ |
|         xkcd:grey pink          | #c3909b | $\fcolorbox{gray}{#c3909b}{\phantom{x}}$ |
|        xkcd:soft purple         | #a66fb5 | $\fcolorbox{gray}{#a66fb5}{\phantom{x}}$ |
|           xkcd:blood            | #770001 | $\fcolorbox{gray}{#770001}{\phantom{x}}$ |
|         xkcd:brown red          | #922b05 | $\fcolorbox{gray}{#922b05}{\phantom{x}}$ |
|        xkcd:medium grey         | #7d7f7c | $\fcolorbox{gray}{#7d7f7c}{\phantom{x}}$ |
|           xkcd:berry            | #990f4b | $\fcolorbox{gray}{#990f4b}{\phantom{x}}$ |
|            xkcd:poo             | #8f7303 | $\fcolorbox{gray}{#8f7303}{\phantom{x}}$ |
|        xkcd:purpley pink        | #c83cb9 | $\fcolorbox{gray}{#c83cb9}{\phantom{x}}$ |
|        xkcd:light salmon        | #fea993 | $\fcolorbox{gray}{#fea993}{\phantom{x}}$ |
|            xkcd:snot            | #acbb0d | $\fcolorbox{gray}{#acbb0d}{\phantom{x}}$ |
|       xkcd:easter purple        | #c071fe | $\fcolorbox{gray}{#c071fe}{\phantom{x}}$ |
|     xkcd:light yellow green     | #ccfd7f | $\fcolorbox{gray}{#ccfd7f}{\phantom{x}}$ |
|       xkcd:dark navy blue       | #00022e | $\fcolorbox{gray}{#00022e}{\phantom{x}}$ |
|            xkcd:drab            | #828344 | $\fcolorbox{gray}{#828344}{\phantom{x}}$ |
|         xkcd:light rose         | #ffc5cb | $\fcolorbox{gray}{#ffc5cb}{\phantom{x}}$ |
|           xkcd:rouge            | #ab1239 | $\fcolorbox{gray}{#ab1239}{\phantom{x}}$ |
|        xkcd:purplish red        | #b0054b | $\fcolorbox{gray}{#b0054b}{\phantom{x}}$ |
|        xkcd:slime green         | #99cc04 | $\fcolorbox{gray}{#99cc04}{\phantom{x}}$ |
|         xkcd:baby poop          | #937c00 | $\fcolorbox{gray}{#937c00}{\phantom{x}}$ |
|        xkcd:irish green         | #019529 | $\fcolorbox{gray}{#019529}{\phantom{x}}$ |
|        xkcd:pink/purple         | #ef1de7 | $\fcolorbox{gray}{#ef1de7}{\phantom{x}}$ |
|         xkcd:dark navy          | #000435 | $\fcolorbox{gray}{#000435}{\phantom{x}}$ |
|        xkcd:greeny blue         | #42b395 | $\fcolorbox{gray}{#42b395}{\phantom{x}}$ |
|         xkcd:light plum         | #9d5783 | $\fcolorbox{gray}{#9d5783}{\phantom{x}}$ |
|        xkcd:pinkish grey        | #c8aca9 | $\fcolorbox{gray}{#c8aca9}{\phantom{x}}$ |
|        xkcd:dirty orange        | #c87606 | $\fcolorbox{gray}{#c87606}{\phantom{x}}$ |
|          xkcd:rust red          | #aa2704 | $\fcolorbox{gray}{#aa2704}{\phantom{x}}$ |
|         xkcd:pale lilac         | #e4cbff | $\fcolorbox{gray}{#e4cbff}{\phantom{x}}$ |
|        xkcd:orangey red         | #fa4224 | $\fcolorbox{gray}{#fa4224}{\phantom{x}}$ |
|        xkcd:primary blue        | #0804f9 | $\fcolorbox{gray}{#0804f9}{\phantom{x}}$ |
|        xkcd:kermit green        | #5cb200 | $\fcolorbox{gray}{#5cb200}{\phantom{x}}$ |
|      xkcd:brownish purple       | #76424e | $\fcolorbox{gray}{#76424e}{\phantom{x}}$ |
|        xkcd:murky green         | #6c7a0e | $\fcolorbox{gray}{#6c7a0e}{\phantom{x}}$ |
|           xkcd:wheat            | #fbdd7e | $\fcolorbox{gray}{#fbdd7e}{\phantom{x}}$ |
|      xkcd:very dark purple      | #2a0134 | $\fcolorbox{gray}{#2a0134}{\phantom{x}}$ |
|        xkcd:bottle green        | #044a05 | $\fcolorbox{gray}{#044a05}{\phantom{x}}$ |
|         xkcd:watermelon         | #fd4659 | $\fcolorbox{gray}{#fd4659}{\phantom{x}}$ |
|       xkcd:deep sky blue        | #0d75f8 | $\fcolorbox{gray}{#0d75f8}{\phantom{x}}$ |
|      xkcd:fire engine red       | #fe0002 | $\fcolorbox{gray}{#fe0002}{\phantom{x}}$ |
|        xkcd:yellow ochre        | #cb9d06 | $\fcolorbox{gray}{#cb9d06}{\phantom{x}}$ |
|       xkcd:pumpkin orange       | #fb7d07 | $\fcolorbox{gray}{#fb7d07}{\phantom{x}}$ |
|         xkcd:pale olive         | #b9cc81 | $\fcolorbox{gray}{#b9cc81}{\phantom{x}}$ |
|        xkcd:light lilac         | #edc8ff | $\fcolorbox{gray}{#edc8ff}{\phantom{x}}$ |
|       xkcd:lightish green       | #61e160 | $\fcolorbox{gray}{#61e160}{\phantom{x}}$ |
|       xkcd:carolina blue        | #8ab8fe | $\fcolorbox{gray}{#8ab8fe}{\phantom{x}}$ |
|          xkcd:mulberry          | #920a4e | $\fcolorbox{gray}{#920a4e}{\phantom{x}}$ |
|       xkcd:shocking pink        | #fe02a2 | $\fcolorbox{gray}{#fe02a2}{\phantom{x}}$ |
|           xkcd:auburn           | #9a3001 | $\fcolorbox{gray}{#9a3001}{\phantom{x}}$ |
|     xkcd:bright lime green      | #65fe08 | $\fcolorbox{gray}{#65fe08}{\phantom{x}}$ |
|          xkcd:celadon           | #befdb7 | $\fcolorbox{gray}{#befdb7}{\phantom{x}}$ |
|       xkcd:pinkish brown        | #b17261 | $\fcolorbox{gray}{#b17261}{\phantom{x}}$ |
|         xkcd:poo brown          | #885f01 | $\fcolorbox{gray}{#885f01}{\phantom{x}}$ |
|      xkcd:bright sky blue       | #02ccfe | $\fcolorbox{gray}{#02ccfe}{\phantom{x}}$ |
|           xkcd:celery           | #c1fd95 | $\fcolorbox{gray}{#c1fd95}{\phantom{x}}$ |
|         xkcd:dirt brown         | #836539 | $\fcolorbox{gray}{#836539}{\phantom{x}}$ |
|         xkcd:strawberry         | #fb2943 | $\fcolorbox{gray}{#fb2943}{\phantom{x}}$ |
|         xkcd:dark lime          | #84b701 | $\fcolorbox{gray}{#84b701}{\phantom{x}}$ |
|           xkcd:copper           | #b66325 | $\fcolorbox{gray}{#b66325}{\phantom{x}}$ |
|        xkcd:medium brown        | #7f5112 | $\fcolorbox{gray}{#7f5112}{\phantom{x}}$ |
|        xkcd:muted green         | #5fa052 | $\fcolorbox{gray}{#5fa052}{\phantom{x}}$ |
|        xkcd:robin's egg         | #6dedfd | $\fcolorbox{gray}{#6dedfd}{\phantom{x}}$ |
|        xkcd:bright aqua         | #0bf9ea | $\fcolorbox{gray}{#0bf9ea}{\phantom{x}}$ |
|      xkcd:bright lavender       | #c760ff | $\fcolorbox{gray}{#c760ff}{\phantom{x}}$ |
|           xkcd:ivory            | #ffffcb | $\fcolorbox{gray}{#ffffcb}{\phantom{x}}$ |
|     xkcd:very light purple      | #f6cefc | $\fcolorbox{gray}{#f6cefc}{\phantom{x}}$ |
|         xkcd:light navy         | #155084 | $\fcolorbox{gray}{#155084}{\phantom{x}}$ |
|          xkcd:pink red          | #f5054f | $\fcolorbox{gray}{#f5054f}{\phantom{x}}$ |
|        xkcd:olive brown         | #645403 | $\fcolorbox{gray}{#645403}{\phantom{x}}$ |
|         xkcd:poop brown         | #7a5901 | $\fcolorbox{gray}{#7a5901}{\phantom{x}}$ |
|       xkcd:mustard green        | #a8b504 | $\fcolorbox{gray}{#a8b504}{\phantom{x}}$ |
|        xkcd:ocean green         | #3d9973 | $\fcolorbox{gray}{#3d9973}{\phantom{x}}$ |
|       xkcd:very dark blue       | #000133 | $\fcolorbox{gray}{#000133}{\phantom{x}}$ |
|        xkcd:dusty green         | #76a973 | $\fcolorbox{gray}{#76a973}{\phantom{x}}$ |
|      xkcd:light navy blue       | #2e5a88 | $\fcolorbox{gray}{#2e5a88}{\phantom{x}}$ |
|        xkcd:minty green         | #0bf77d | $\fcolorbox{gray}{#0bf77d}{\phantom{x}}$ |
|           xkcd:adobe            | #bd6c48 | $\fcolorbox{gray}{#bd6c48}{\phantom{x}}$ |
|           xkcd:barney           | #ac1db8 | $\fcolorbox{gray}{#ac1db8}{\phantom{x}}$ |
|         xkcd:jade green         | #2baf6a | $\fcolorbox{gray}{#2baf6a}{\phantom{x}}$ |
|     xkcd:bright light blue      | #26f7fd | $\fcolorbox{gray}{#26f7fd}{\phantom{x}}$ |
|         xkcd:light lime         | #aefd6c | $\fcolorbox{gray}{#aefd6c}{\phantom{x}}$ |
|         xkcd:dark khaki         | #9b8f55 | $\fcolorbox{gray}{#9b8f55}{\phantom{x}}$ |
|       xkcd:orange yellow        | #ffad01 | $\fcolorbox{gray}{#ffad01}{\phantom{x}}$ |
|            xkcd:ocre            | #c69c04 | $\fcolorbox{gray}{#c69c04}{\phantom{x}}$ |
|           xkcd:maize            | #f4d054 | $\fcolorbox{gray}{#f4d054}{\phantom{x}}$ |
|         xkcd:faded pink         | #de9dac | $\fcolorbox{gray}{#de9dac}{\phantom{x}}$ |
|    xkcd:british racing green    | #05480d | $\fcolorbox{gray}{#05480d}{\phantom{x}}$ |
|         xkcd:sandstone          | #c9ae74 | $\fcolorbox{gray}{#c9ae74}{\phantom{x}}$ |
|         xkcd:mud brown          | #60460f | $\fcolorbox{gray}{#60460f}{\phantom{x}}$ |
|      xkcd:light sea green       | #98f6b0 | $\fcolorbox{gray}{#98f6b0}{\phantom{x}}$ |
|       xkcd:robin egg blue       | #8af1fe | $\fcolorbox{gray}{#8af1fe}{\phantom{x}}$ |
|        xkcd:aqua marine         | #2ee8bb | $\fcolorbox{gray}{#2ee8bb}{\phantom{x}}$ |
|       xkcd:dark sea green       | #11875d | $\fcolorbox{gray}{#11875d}{\phantom{x}}$ |
|         xkcd:soft pink          | #fdb0c0 | $\fcolorbox{gray}{#fdb0c0}{\phantom{x}}$ |
|       xkcd:orangey brown        | #b16002 | $\fcolorbox{gray}{#b16002}{\phantom{x}}$ |
|         xkcd:cherry red         | #f7022a | $\fcolorbox{gray}{#f7022a}{\phantom{x}}$ |
|        xkcd:burnt yellow        | #d5ab09 | $\fcolorbox{gray}{#d5ab09}{\phantom{x}}$ |
|       xkcd:brownish grey        | #86775f | $\fcolorbox{gray}{#86775f}{\phantom{x}}$ |
|           xkcd:camel            | #c69f59 | $\fcolorbox{gray}{#c69f59}{\phantom{x}}$ |
|       xkcd:purplish grey        | #7a687f | $\fcolorbox{gray}{#7a687f}{\phantom{x}}$ |
|           xkcd:marine           | #042e60 | $\fcolorbox{gray}{#042e60}{\phantom{x}}$ |
|        xkcd:greyish pink        | #c88d94 | $\fcolorbox{gray}{#c88d94}{\phantom{x}}$ |
|       xkcd:pale turquoise       | #a5fbd5 | $\fcolorbox{gray}{#a5fbd5}{\phantom{x}}$ |
|       xkcd:pastel yellow        | #fffe71 | $\fcolorbox{gray}{#fffe71}{\phantom{x}}$ |
|        xkcd:bluey purple        | #6241c7 | $\fcolorbox{gray}{#6241c7}{\phantom{x}}$ |
|       xkcd:canary yellow        | #fffe40 | $\fcolorbox{gray}{#fffe40}{\phantom{x}}$ |
|         xkcd:faded red          | #d3494e | $\fcolorbox{gray}{#d3494e}{\phantom{x}}$ |
|           xkcd:sepia            | #985e2b | $\fcolorbox{gray}{#985e2b}{\phantom{x}}$ |
|           xkcd:coffee           | #a6814c | $\fcolorbox{gray}{#a6814c}{\phantom{x}}$ |
|       xkcd:bright magenta       | #ff08e8 | $\fcolorbox{gray}{#ff08e8}{\phantom{x}}$ |
|           xkcd:mocha            | #9d7651 | $\fcolorbox{gray}{#9d7651}{\phantom{x}}$ |
|            xkcd:ecru            | #feffca | $\fcolorbox{gray}{#feffca}{\phantom{x}}$ |
|         xkcd:purpleish          | #98568d | $\fcolorbox{gray}{#98568d}{\phantom{x}}$ |
|         xkcd:cranberry          | #9e003a | $\fcolorbox{gray}{#9e003a}{\phantom{x}}$ |
|       xkcd:darkish green        | #287c37 | $\fcolorbox{gray}{#287c37}{\phantom{x}}$ |
|        xkcd:brown orange        | #b96902 | $\fcolorbox{gray}{#b96902}{\phantom{x}}$ |
|         xkcd:dusky rose         | #ba6873 | $\fcolorbox{gray}{#ba6873}{\phantom{x}}$ |
|           xkcd:melon            | #ff7855 | $\fcolorbox{gray}{#ff7855}{\phantom{x}}$ |
|        xkcd:sickly green        | #94b21c | $\fcolorbox{gray}{#94b21c}{\phantom{x}}$ |
|           xkcd:silver           | #c5c9c7 | $\fcolorbox{gray}{#c5c9c7}{\phantom{x}}$ |
|        xkcd:purply blue         | #661aee | $\fcolorbox{gray}{#661aee}{\phantom{x}}$ |
|       xkcd:purpleish blue       | #6140ef | $\fcolorbox{gray}{#6140ef}{\phantom{x}}$ |
|       xkcd:hospital green       | #9be5aa | $\fcolorbox{gray}{#9be5aa}{\phantom{x}}$ |
|         xkcd:shit brown         | #7b5804 | $\fcolorbox{gray}{#7b5804}{\phantom{x}}$ |
|          xkcd:mid blue          | #276ab3 | $\fcolorbox{gray}{#276ab3}{\phantom{x}}$ |
|           xkcd:amber            | #feb308 | $\fcolorbox{gray}{#feb308}{\phantom{x}}$ |
|        xkcd:easter green        | #8cfd7e | $\fcolorbox{gray}{#8cfd7e}{\phantom{x}}$ |
|         xkcd:soft blue          | #6488ea | $\fcolorbox{gray}{#6488ea}{\phantom{x}}$ |
|       xkcd:cerulean blue        | #056eee | $\fcolorbox{gray}{#056eee}{\phantom{x}}$ |
|        xkcd:golden brown        | #b27a01 | $\fcolorbox{gray}{#b27a01}{\phantom{x}}$ |
|      xkcd:bright turquoise      | #0ffef9 | $\fcolorbox{gray}{#0ffef9}{\phantom{x}}$ |
|          xkcd:red pink          | #fa2a55 | $\fcolorbox{gray}{#fa2a55}{\phantom{x}}$ |
|         xkcd:red purple         | #820747 | $\fcolorbox{gray}{#820747}{\phantom{x}}$ |
|       xkcd:greyish brown        | #7a6a4f | $\fcolorbox{gray}{#7a6a4f}{\phantom{x}}$ |
|         xkcd:vermillion         | #f4320c | $\fcolorbox{gray}{#f4320c}{\phantom{x}}$ |
|           xkcd:russet           | #a13905 | $\fcolorbox{gray}{#a13905}{\phantom{x}}$ |
|         xkcd:steel grey         | #6f828a | $\fcolorbox{gray}{#6f828a}{\phantom{x}}$ |
|       xkcd:lighter purple       | #a55af4 | $\fcolorbox{gray}{#a55af4}{\phantom{x}}$ |
|       xkcd:bright violet        | #ad0afd | $\fcolorbox{gray}{#ad0afd}{\phantom{x}}$ |
|       xkcd:prussian blue        | #004577 | $\fcolorbox{gray}{#004577}{\phantom{x}}$ |
|        xkcd:slate green         | #658d6d | $\fcolorbox{gray}{#658d6d}{\phantom{x}}$ |
|         xkcd:dirty pink         | #ca7b80 | $\fcolorbox{gray}{#ca7b80}{\phantom{x}}$ |
|      xkcd:dark blue green       | #005249 | $\fcolorbox{gray}{#005249}{\phantom{x}}$ |
|            xkcd:pine            | #2b5d34 | $\fcolorbox{gray}{#2b5d34}{\phantom{x}}$ |
|       xkcd:yellowy green        | #bff128 | $\fcolorbox{gray}{#bff128}{\phantom{x}}$ |
|         xkcd:dark gold          | #b59410 | $\fcolorbox{gray}{#b59410}{\phantom{x}}$ |
|           xkcd:bluish           | #2976bb | $\fcolorbox{gray}{#2976bb}{\phantom{x}}$ |
|        xkcd:darkish blue        | #014182 | $\fcolorbox{gray}{#014182}{\phantom{x}}$ |
|          xkcd:dull red          | #bb3f3f | $\fcolorbox{gray}{#bb3f3f}{\phantom{x}}$ |
|         xkcd:pinky red          | #fc2647 | $\fcolorbox{gray}{#fc2647}{\phantom{x}}$ |
|           xkcd:bronze           | #a87900 | $\fcolorbox{gray}{#a87900}{\phantom{x}}$ |
|         xkcd:pale teal          | #82cbb2 | $\fcolorbox{gray}{#82cbb2}{\phantom{x}}$ |
|       xkcd:military green       | #667c3e | $\fcolorbox{gray}{#667c3e}{\phantom{x}}$ |
|        xkcd:barbie pink         | #fe46a5 | $\fcolorbox{gray}{#fe46a5}{\phantom{x}}$ |
|       xkcd:bubblegum pink       | #fe83cc | $\fcolorbox{gray}{#fe83cc}{\phantom{x}}$ |
|       xkcd:pea soup green       | #94a617 | $\fcolorbox{gray}{#94a617}{\phantom{x}}$ |
|        xkcd:dark mustard        | #a88905 | $\fcolorbox{gray}{#a88905}{\phantom{x}}$ |
|            xkcd:shit            | #7f5f00 | $\fcolorbox{gray}{#7f5f00}{\phantom{x}}$ |
|       xkcd:medium purple        | #9e43a2 | $\fcolorbox{gray}{#9e43a2}{\phantom{x}}$ |
|      xkcd:very dark green       | #062e03 | $\fcolorbox{gray}{#062e03}{\phantom{x}}$ |
|            xkcd:dirt            | #8a6e45 | $\fcolorbox{gray}{#8a6e45}{\phantom{x}}$ |
|         xkcd:dusky pink         | #cc7a8b | $\fcolorbox{gray}{#cc7a8b}{\phantom{x}}$ |
|         xkcd:red violet         | #9e0168 | $\fcolorbox{gray}{#9e0168}{\phantom{x}}$ |
|        xkcd:lemon yellow        | #fdff38 | $\fcolorbox{gray}{#fdff38}{\phantom{x}}$ |
|         xkcd:pistachio          | #c0fa8b | $\fcolorbox{gray}{#c0fa8b}{\phantom{x}}$ |
|        xkcd:dull yellow         | #eedc5b | $\fcolorbox{gray}{#eedc5b}{\phantom{x}}$ |
|      xkcd:dark lime green       | #7ebd01 | $\fcolorbox{gray}{#7ebd01}{\phantom{x}}$ |
|         xkcd:denim blue         | #3b5b92 | $\fcolorbox{gray}{#3b5b92}{\phantom{x}}$ |
|         xkcd:teal blue          | #01889f | $\fcolorbox{gray}{#01889f}{\phantom{x}}$ |
|       xkcd:lightish blue        | #3d7afd | $\fcolorbox{gray}{#3d7afd}{\phantom{x}}$ |
|        xkcd:purpley blue        | #5f34e7 | $\fcolorbox{gray}{#5f34e7}{\phantom{x}}$ |
|        xkcd:light indigo        | #6d5acf | $\fcolorbox{gray}{#6d5acf}{\phantom{x}}$ |
|        xkcd:swamp green         | #748500 | $\fcolorbox{gray}{#748500}{\phantom{x}}$ |
|        xkcd:brown green         | #706c11 | $\fcolorbox{gray}{#706c11}{\phantom{x}}$ |
|        xkcd:dark maroon         | #3c0008 | $\fcolorbox{gray}{#3c0008}{\phantom{x}}$ |
|         xkcd:hot purple         | #cb00f5 | $\fcolorbox{gray}{#cb00f5}{\phantom{x}}$ |
|     xkcd:dark forest green      | #002d04 | $\fcolorbox{gray}{#002d04}{\phantom{x}}$ |
|         xkcd:faded blue         | #658cbb | $\fcolorbox{gray}{#658cbb}{\phantom{x}}$ |
|         xkcd:drab green         | #749551 | $\fcolorbox{gray}{#749551}{\phantom{x}}$ |
|      xkcd:light lime green      | #b9ff66 | $\fcolorbox{gray}{#b9ff66}{\phantom{x}}$ |
|         xkcd:snot green         | #9dc100 | $\fcolorbox{gray}{#9dc100}{\phantom{x}}$ |
|         xkcd:yellowish          | #faee66 | $\fcolorbox{gray}{#faee66}{\phantom{x}}$ |
|      xkcd:light blue green      | #7efbb3 | $\fcolorbox{gray}{#7efbb3}{\phantom{x}}$ |
|          xkcd:bordeaux          | #7b002c | $\fcolorbox{gray}{#7b002c}{\phantom{x}}$ |
|        xkcd:light mauve         | #c292a1 | $\fcolorbox{gray}{#c292a1}{\phantom{x}}$ |
|           xkcd:ocean            | #017b92 | $\fcolorbox{gray}{#017b92}{\phantom{x}}$ |
|          xkcd:marigold          | #fcc006 | $\fcolorbox{gray}{#fcc006}{\phantom{x}}$ |
|        xkcd:muddy green         | #657432 | $\fcolorbox{gray}{#657432}{\phantom{x}}$ |
|        xkcd:dull orange         | #d8863b | $\fcolorbox{gray}{#d8863b}{\phantom{x}}$ |
|           xkcd:steel            | #738595 | $\fcolorbox{gray}{#738595}{\phantom{x}}$ |
|      xkcd:electric purple       | #aa23ff | $\fcolorbox{gray}{#aa23ff}{\phantom{x}}$ |
|     xkcd:fluorescent green      | #08ff08 | $\fcolorbox{gray}{#08ff08}{\phantom{x}}$ |
|      xkcd:yellowish brown       | #9b7a01 | $\fcolorbox{gray}{#9b7a01}{\phantom{x}}$ |
|           xkcd:blush            | #f29e8e | $\fcolorbox{gray}{#f29e8e}{\phantom{x}}$ |
|         xkcd:soft green         | #6fc276 | $\fcolorbox{gray}{#6fc276}{\phantom{x}}$ |
|       xkcd:bright orange        | #ff5b00 | $\fcolorbox{gray}{#ff5b00}{\phantom{x}}$ |
|           xkcd:lemon            | #fdff52 | $\fcolorbox{gray}{#fdff52}{\phantom{x}}$ |
|        xkcd:purple grey         | #866f85 | $\fcolorbox{gray}{#866f85}{\phantom{x}}$ |
|         xkcd:acid green         | #8ffe09 | $\fcolorbox{gray}{#8ffe09}{\phantom{x}}$ |
|       xkcd:pale lavender        | #eecffe | $\fcolorbox{gray}{#eecffe}{\phantom{x}}$ |
|        xkcd:violet blue         | #510ac9 | $\fcolorbox{gray}{#510ac9}{\phantom{x}}$ |
|     xkcd:light forest green     | #4f9153 | $\fcolorbox{gray}{#4f9153}{\phantom{x}}$ |
|         xkcd:burnt red          | #9f2305 | $\fcolorbox{gray}{#9f2305}{\phantom{x}}$ |
|        xkcd:khaki green         | #728639 | $\fcolorbox{gray}{#728639}{\phantom{x}}$ |
|           xkcd:cerise           | #de0c62 | $\fcolorbox{gray}{#de0c62}{\phantom{x}}$ |
|        xkcd:faded purple        | #916e99 | $\fcolorbox{gray}{#916e99}{\phantom{x}}$ |
|          xkcd:apricot           | #ffb16d | $\fcolorbox{gray}{#ffb16d}{\phantom{x}}$ |
|      xkcd:dark olive green      | #3c4d03 | $\fcolorbox{gray}{#3c4d03}{\phantom{x}}$ |
|         xkcd:grey brown         | #7f7053 | $\fcolorbox{gray}{#7f7053}{\phantom{x}}$ |
|         xkcd:green grey         | #77926f | $\fcolorbox{gray}{#77926f}{\phantom{x}}$ |
|         xkcd:true blue          | #010fcc | $\fcolorbox{gray}{#010fcc}{\phantom{x}}$ |
|        xkcd:pale violet         | #ceaefa | $\fcolorbox{gray}{#ceaefa}{\phantom{x}}$ |
|      xkcd:periwinkle blue       | #8f99fb | $\fcolorbox{gray}{#8f99fb}{\phantom{x}}$ |
|       xkcd:light sky blue       | #c6fcff | $\fcolorbox{gray}{#c6fcff}{\phantom{x}}$ |
|          xkcd:blurple           | #5539cc | $\fcolorbox{gray}{#5539cc}{\phantom{x}}$ |
|        xkcd:green brown         | #544e03 | $\fcolorbox{gray}{#544e03}{\phantom{x}}$ |
|         xkcd:bluegreen          | #017a79 | $\fcolorbox{gray}{#017a79}{\phantom{x}}$ |
|        xkcd:bright teal         | #01f9c6 | $\fcolorbox{gray}{#01f9c6}{\phantom{x}}$ |
|      xkcd:brownish yellow       | #c9b003 | $\fcolorbox{gray}{#c9b003}{\phantom{x}}$ |
|          xkcd:pea soup          | #929901 | $\fcolorbox{gray}{#929901}{\phantom{x}}$ |
|           xkcd:forest           | #0b5509 | $\fcolorbox{gray}{#0b5509}{\phantom{x}}$ |
|       xkcd:barney purple        | #a00498 | $\fcolorbox{gray}{#a00498}{\phantom{x}}$ |
|        xkcd:ultramarine         | #2000b1 | $\fcolorbox{gray}{#2000b1}{\phantom{x}}$ |
|          xkcd:purplish          | #94568c | $\fcolorbox{gray}{#94568c}{\phantom{x}}$ |
|        xkcd:puke yellow         | #c2be0e | $\fcolorbox{gray}{#c2be0e}{\phantom{x}}$ |
|        xkcd:bluish grey         | #748b97 | $\fcolorbox{gray}{#748b97}{\phantom{x}}$ |
|      xkcd:dark periwinkle       | #665fd1 | $\fcolorbox{gray}{#665fd1}{\phantom{x}}$ |
|         xkcd:dark lilac         | #9c6da5 | $\fcolorbox{gray}{#9c6da5}{\phantom{x}}$ |
|          xkcd:reddish           | #c44240 | $\fcolorbox{gray}{#c44240}{\phantom{x}}$ |
|        xkcd:light maroon        | #a24857 | $\fcolorbox{gray}{#a24857}{\phantom{x}}$ |
|        xkcd:dusty purple        | #825f87 | $\fcolorbox{gray}{#825f87}{\phantom{x}}$ |
|        xkcd:terra cotta         | #c9643b | $\fcolorbox{gray}{#c9643b}{\phantom{x}}$ |
|          xkcd:avocado           | #90b134 | $\fcolorbox{gray}{#90b134}{\phantom{x}}$ |
|        xkcd:marine blue         | #01386a | $\fcolorbox{gray}{#01386a}{\phantom{x}}$ |
|         xkcd:teal green         | #25a36f | $\fcolorbox{gray}{#25a36f}{\phantom{x}}$ |
|         xkcd:slate grey         | #59656d | $\fcolorbox{gray}{#59656d}{\phantom{x}}$ |
|       xkcd:lighter green        | #75fd63 | $\fcolorbox{gray}{#75fd63}{\phantom{x}}$ |
|       xkcd:electric green       | #21fc0d | $\fcolorbox{gray}{#21fc0d}{\phantom{x}}$ |
|         xkcd:dusty blue         | #5a86ad | $\fcolorbox{gray}{#5a86ad}{\phantom{x}}$ |
|       xkcd:golden yellow        | #fec615 | $\fcolorbox{gray}{#fec615}{\phantom{x}}$ |
|       xkcd:bright yellow        | #fffd01 | $\fcolorbox{gray}{#fffd01}{\phantom{x}}$ |
|       xkcd:light lavender       | #dfc5fe | $\fcolorbox{gray}{#dfc5fe}{\phantom{x}}$ |
|           xkcd:umber            | #b26400 | $\fcolorbox{gray}{#b26400}{\phantom{x}}$ |
|            xkcd:poop            | #7f5e00 | $\fcolorbox{gray}{#7f5e00}{\phantom{x}}$ |
|         xkcd:dark peach         | #de7e5d | $\fcolorbox{gray}{#de7e5d}{\phantom{x}}$ |
|        xkcd:jungle green        | #048243 | $\fcolorbox{gray}{#048243}{\phantom{x}}$ |
|          xkcd:eggshell          | #ffffd4 | $\fcolorbox{gray}{#ffffd4}{\phantom{x}}$ |
|           xkcd:denim            | #3b638c | $\fcolorbox{gray}{#3b638c}{\phantom{x}}$ |
|        xkcd:yellow brown        | #b79400 | $\fcolorbox{gray}{#b79400}{\phantom{x}}$ |
|        xkcd:dull purple         | #84597e | $\fcolorbox{gray}{#84597e}{\phantom{x}}$ |
|      xkcd:chocolate brown       | #411900 | $\fcolorbox{gray}{#411900}{\phantom{x}}$ |
|          xkcd:wine red          | #7b0323 | $\fcolorbox{gray}{#7b0323}{\phantom{x}}$ |
|         xkcd:neon blue          | #04d9ff | $\fcolorbox{gray}{#04d9ff}{\phantom{x}}$ |
|        xkcd:dirty green         | #667e2c | $\fcolorbox{gray}{#667e2c}{\phantom{x}}$ |
|         xkcd:light tan          | #fbeeac | $\fcolorbox{gray}{#fbeeac}{\phantom{x}}$ |
|          xkcd:ice blue          | #d7fffe | $\fcolorbox{gray}{#d7fffe}{\phantom{x}}$ |
|         xkcd:cadet blue         | #4e7496 | $\fcolorbox{gray}{#4e7496}{\phantom{x}}$ |
|         xkcd:dark mauve         | #874c62 | $\fcolorbox{gray}{#874c62}{\phantom{x}}$ |
|      xkcd:very light blue       | #d5ffff | $\fcolorbox{gray}{#d5ffff}{\phantom{x}}$ |
|        xkcd:grey purple         | #826d8c | $\fcolorbox{gray}{#826d8c}{\phantom{x}}$ |
|        xkcd:pastel pink         | #ffbacd | $\fcolorbox{gray}{#ffbacd}{\phantom{x}}$ |
|      xkcd:very light green      | #d1ffbd | $\fcolorbox{gray}{#d1ffbd}{\phantom{x}}$ |
|       xkcd:dark sky blue        | #448ee4 | $\fcolorbox{gray}{#448ee4}{\phantom{x}}$ |
|         xkcd:evergreen          | #05472a | $\fcolorbox{gray}{#05472a}{\phantom{x}}$ |
|         xkcd:dull pink          | #d5869d | $\fcolorbox{gray}{#d5869d}{\phantom{x}}$ |
|         xkcd:aubergine          | #3d0734 | $\fcolorbox{gray}{#3d0734}{\phantom{x}}$ |
|          xkcd:mahogany          | #4a0100 | $\fcolorbox{gray}{#4a0100}{\phantom{x}}$ |
|       xkcd:reddish orange       | #f8481c | $\fcolorbox{gray}{#f8481c}{\phantom{x}}$ |
|         xkcd:deep green         | #02590f | $\fcolorbox{gray}{#02590f}{\phantom{x}}$ |
|        xkcd:vomit green         | #89a203 | $\fcolorbox{gray}{#89a203}{\phantom{x}}$ |
|        xkcd:purple pink         | #e03fd8 | $\fcolorbox{gray}{#e03fd8}{\phantom{x}}$ |
|         xkcd:dusty pink         | #d58a94 | $\fcolorbox{gray}{#d58a94}{\phantom{x}}$ |
|        xkcd:faded green         | #7bb274 | $\fcolorbox{gray}{#7bb274}{\phantom{x}}$ |
|         xkcd:camo green         | #526525 | $\fcolorbox{gray}{#526525}{\phantom{x}}$ |
|        xkcd:pinky purple        | #c94cbe | $\fcolorbox{gray}{#c94cbe}{\phantom{x}}$ |
|        xkcd:pink purple         | #db4bda | $\fcolorbox{gray}{#db4bda}{\phantom{x}}$ |
|        xkcd:brownish red        | #9e3623 | $\fcolorbox{gray}{#9e3623}{\phantom{x}}$ |
|         xkcd:dark rose          | #b5485d | $\fcolorbox{gray}{#b5485d}{\phantom{x}}$ |
|            xkcd:mud             | #735c12 | $\fcolorbox{gray}{#735c12}{\phantom{x}}$ |
|          xkcd:brownish          | #9c6d57 | $\fcolorbox{gray}{#9c6d57}{\phantom{x}}$ |
|       xkcd:emerald green        | #028f1e | $\fcolorbox{gray}{#028f1e}{\phantom{x}}$ |
|         xkcd:pale brown         | #b1916e | $\fcolorbox{gray}{#b1916e}{\phantom{x}}$ |
|         xkcd:dull blue          | #49759c | $\fcolorbox{gray}{#49759c}{\phantom{x}}$ |
|        xkcd:burnt umber         | #a0450e | $\fcolorbox{gray}{#a0450e}{\phantom{x}}$ |
|        xkcd:medium green        | #39ad48 | $\fcolorbox{gray}{#39ad48}{\phantom{x}}$ |
|            xkcd:clay            | #b66a50 | $\fcolorbox{gray}{#b66a50}{\phantom{x}}$ |
|         xkcd:light aqua         | #8cffdb | $\fcolorbox{gray}{#8cffdb}{\phantom{x}}$ |
|     xkcd:light olive green      | #a4be5c | $\fcolorbox{gray}{#a4be5c}{\phantom{x}}$ |
|      xkcd:brownish orange       | #cb7723 | $\fcolorbox{gray}{#cb7723}{\phantom{x}}$ |
|         xkcd:dark aqua          | #05696b | $\fcolorbox{gray}{#05696b}{\phantom{x}}$ |
|       xkcd:purplish pink        | #ce5dae | $\fcolorbox{gray}{#ce5dae}{\phantom{x}}$ |
|        xkcd:dark salmon         | #c85a53 | $\fcolorbox{gray}{#c85a53}{\phantom{x}}$ |
|       xkcd:greenish grey        | #96ae8d | $\fcolorbox{gray}{#96ae8d}{\phantom{x}}$ |
|            xkcd:jade            | #1fa774 | $\fcolorbox{gray}{#1fa774}{\phantom{x}}$ |
|         xkcd:ugly green         | #7a9703 | $\fcolorbox{gray}{#7a9703}{\phantom{x}}$ |
|         xkcd:dark beige         | #ac9362 | $\fcolorbox{gray}{#ac9362}{\phantom{x}}$ |
|          xkcd:emerald           | #01a049 | $\fcolorbox{gray}{#01a049}{\phantom{x}}$ |
|          xkcd:pale red          | #d9544d | $\fcolorbox{gray}{#d9544d}{\phantom{x}}$ |
|       xkcd:light magenta        | #fa5ff7 | $\fcolorbox{gray}{#fa5ff7}{\phantom{x}}$ |
|            xkcd:sky             | #82cafc | $\fcolorbox{gray}{#82cafc}{\phantom{x}}$ |
|         xkcd:light cyan         | #acfffc | $\fcolorbox{gray}{#acfffc}{\phantom{x}}$ |
|       xkcd:yellow orange        | #fcb001 | $\fcolorbox{gray}{#fcb001}{\phantom{x}}$ |
|       xkcd:reddish purple       | #910951 | $\fcolorbox{gray}{#910951}{\phantom{x}}$ |
|        xkcd:reddish pink        | #fe2c54 | $\fcolorbox{gray}{#fe2c54}{\phantom{x}}$ |
|           xkcd:orchid           | #c875c4 | $\fcolorbox{gray}{#c875c4}{\phantom{x}}$ |
|        xkcd:dirty yellow        | #cdc50a | $\fcolorbox{gray}{#cdc50a}{\phantom{x}}$ |
|         xkcd:orange red         | #fd411e | $\fcolorbox{gray}{#fd411e}{\phantom{x}}$ |
|          xkcd:deep red          | #9a0200 | $\fcolorbox{gray}{#9a0200}{\phantom{x}}$ |
|        xkcd:orange brown        | #be6400 | $\fcolorbox{gray}{#be6400}{\phantom{x}}$ |
|        xkcd:cobalt blue         | #030aa7 | $\fcolorbox{gray}{#030aa7}{\phantom{x}}$ |
|         xkcd:neon pink          | #fe019a | $\fcolorbox{gray}{#fe019a}{\phantom{x}}$ |
|         xkcd:rose pink          | #f7879a | $\fcolorbox{gray}{#f7879a}{\phantom{x}}$ |
|       xkcd:greyish purple       | #887191 | $\fcolorbox{gray}{#887191}{\phantom{x}}$ |
|         xkcd:raspberry          | #b00149 | $\fcolorbox{gray}{#b00149}{\phantom{x}}$ |
|         xkcd:aqua green         | #12e193 | $\fcolorbox{gray}{#12e193}{\phantom{x}}$ |
|        xkcd:salmon pink         | #fe7b7c | $\fcolorbox{gray}{#fe7b7c}{\phantom{x}}$ |
|         xkcd:tangerine          | #ff9408 | $\fcolorbox{gray}{#ff9408}{\phantom{x}}$ |
|       xkcd:brownish green       | #6a6e09 | $\fcolorbox{gray}{#6a6e09}{\phantom{x}}$ |
|         xkcd:red brown          | #8b2e16 | $\fcolorbox{gray}{#8b2e16}{\phantom{x}}$ |
|       xkcd:greenish brown       | #696112 | $\fcolorbox{gray}{#696112}{\phantom{x}}$ |
|          xkcd:pumpkin           | #e17701 | $\fcolorbox{gray}{#e17701}{\phantom{x}}$ |
|         xkcd:pine green         | #0a481e | $\fcolorbox{gray}{#0a481e}{\phantom{x}}$ |
|          xkcd:charcoal          | #343837 | $\fcolorbox{gray}{#343837}{\phantom{x}}$ |
|         xkcd:baby pink          | #ffb7ce | $\fcolorbox{gray}{#ffb7ce}{\phantom{x}}$ |
|         xkcd:cornflower         | #6a79f7 | $\fcolorbox{gray}{#6a79f7}{\phantom{x}}$ |
|        xkcd:blue violet         | #5d06e9 | $\fcolorbox{gray}{#5d06e9}{\phantom{x}}$ |
|         xkcd:chocolate          | #3d1c02 | $\fcolorbox{gray}{#3d1c02}{\phantom{x}}$ |
|       xkcd:greyish green        | #82a67d | $\fcolorbox{gray}{#82a67d}{\phantom{x}}$ |
|          xkcd:scarlet           | #be0119 | $\fcolorbox{gray}{#be0119}{\phantom{x}}$ |
|        xkcd:green yellow        | #c9ff27 | $\fcolorbox{gray}{#c9ff27}{\phantom{x}}$ |
|         xkcd:dark olive         | #373e02 | $\fcolorbox{gray}{#373e02}{\phantom{x}}$ |
|           xkcd:sienna           | #a9561e | $\fcolorbox{gray}{#a9561e}{\phantom{x}}$ |
|       xkcd:pastel purple        | #caa0ff | $\fcolorbox{gray}{#caa0ff}{\phantom{x}}$ |
|         xkcd:terracotta         | #ca6641 | $\fcolorbox{gray}{#ca6641}{\phantom{x}}$ |
|         xkcd:aqua blue          | #02d8e9 | $\fcolorbox{gray}{#02d8e9}{\phantom{x}}$ |
|         xkcd:sage green         | #88b378 | $\fcolorbox{gray}{#88b378}{\phantom{x}}$ |
|         xkcd:blood red          | #980002 | $\fcolorbox{gray}{#980002}{\phantom{x}}$ |
|         xkcd:deep pink          | #cb0162 | $\fcolorbox{gray}{#cb0162}{\phantom{x}}$ |
|           xkcd:grass            | #5cac2d | $\fcolorbox{gray}{#5cac2d}{\phantom{x}}$ |
|            xkcd:moss            | #769958 | $\fcolorbox{gray}{#769958}{\phantom{x}}$ |
|        xkcd:pastel blue         | #a2bffe | $\fcolorbox{gray}{#a2bffe}{\phantom{x}}$ |
|        xkcd:bluish green        | #10a674 | $\fcolorbox{gray}{#10a674}{\phantom{x}}$ |
|         xkcd:green blue         | #06b48b | $\fcolorbox{gray}{#06b48b}{\phantom{x}}$ |
|          xkcd:dark tan          | #af884a | $\fcolorbox{gray}{#af884a}{\phantom{x}}$ |
|       xkcd:greenish blue        | #0b8b87 | $\fcolorbox{gray}{#0b8b87}{\phantom{x}}$ |
|        xkcd:pale orange         | #ffa756 | $\fcolorbox{gray}{#ffa756}{\phantom{x}}$ |
|           xkcd:vomit            | #a2a415 | $\fcolorbox{gray}{#a2a415}{\phantom{x}}$ |
|       xkcd:forrest green        | #154406 | $\fcolorbox{gray}{#154406}{\phantom{x}}$ |
|       xkcd:dark lavender        | #856798 | $\fcolorbox{gray}{#856798}{\phantom{x}}$ |
|        xkcd:dark violet         | #34013f | $\fcolorbox{gray}{#34013f}{\phantom{x}}$ |
|        xkcd:purple blue         | #632de9 | $\fcolorbox{gray}{#632de9}{\phantom{x}}$ |
|         xkcd:dark cyan          | #0a888a | $\fcolorbox{gray}{#0a888a}{\phantom{x}}$ |
|         xkcd:olive drab         | #6f7632 | $\fcolorbox{gray}{#6f7632}{\phantom{x}}$ |
|          xkcd:pinkish           | #d46a7e | $\fcolorbox{gray}{#d46a7e}{\phantom{x}}$ |
|           xkcd:cobalt           | #1e488f | $\fcolorbox{gray}{#1e488f}{\phantom{x}}$ |
|        xkcd:neon purple         | #bc13fe | $\fcolorbox{gray}{#bc13fe}{\phantom{x}}$ |
|      xkcd:light turquoise       | #7ef4cc | $\fcolorbox{gray}{#7ef4cc}{\phantom{x}}$ |
|        xkcd:apple green         | #76cd26 | $\fcolorbox{gray}{#76cd26}{\phantom{x}}$ |
|         xkcd:dull green         | #74a662 | $\fcolorbox{gray}{#74a662}{\phantom{x}}$ |
|            xkcd:wine            | #80013f | $\fcolorbox{gray}{#80013f}{\phantom{x}}$ |
|        xkcd:powder blue         | #b1d1fc | $\fcolorbox{gray}{#b1d1fc}{\phantom{x}}$ |
|         xkcd:off white          | #ffffe4 | $\fcolorbox{gray}{#ffffe4}{\phantom{x}}$ |
|       xkcd:electric blue        | #0652ff | $\fcolorbox{gray}{#0652ff}{\phantom{x}}$ |
|       xkcd:dark turquoise       | #045c5a | $\fcolorbox{gray}{#045c5a}{\phantom{x}}$ |
|        xkcd:blue purple         | #5729ce | $\fcolorbox{gray}{#5729ce}{\phantom{x}}$ |
|           xkcd:azure            | #069af3 | $\fcolorbox{gray}{#069af3}{\phantom{x}}$ |
|         xkcd:bright red         | #ff000d | $\fcolorbox{gray}{#ff000d}{\phantom{x}}$ |
|        xkcd:pinkish red         | #f10c45 | $\fcolorbox{gray}{#f10c45}{\phantom{x}}$ |
|      xkcd:cornflower blue       | #5170d7 | $\fcolorbox{gray}{#5170d7}{\phantom{x}}$ |
|        xkcd:light olive         | #acbf69 | $\fcolorbox{gray}{#acbf69}{\phantom{x}}$ |
|           xkcd:grape            | #6c3461 | $\fcolorbox{gray}{#6c3461}{\phantom{x}}$ |
|        xkcd:greyish blue        | #5e819d | $\fcolorbox{gray}{#5e819d}{\phantom{x}}$ |
|       xkcd:purplish blue        | #601ef9 | $\fcolorbox{gray}{#601ef9}{\phantom{x}}$ |
|      xkcd:yellowish green       | #b0dd16 | $\fcolorbox{gray}{#b0dd16}{\phantom{x}}$ |
|      xkcd:greenish yellow       | #cdfd02 | $\fcolorbox{gray}{#cdfd02}{\phantom{x}}$ |
|        xkcd:medium blue         | #2c6fbb | $\fcolorbox{gray}{#2c6fbb}{\phantom{x}}$ |
|         xkcd:dusty rose         | #c0737a | $\fcolorbox{gray}{#c0737a}{\phantom{x}}$ |
|        xkcd:light violet        | #d6b4fc | $\fcolorbox{gray}{#d6b4fc}{\phantom{x}}$ |
|       xkcd:midnight blue        | #020035 | $\fcolorbox{gray}{#020035}{\phantom{x}}$ |
|       xkcd:bluish purple        | #703be7 | $\fcolorbox{gray}{#703be7}{\phantom{x}}$ |
|         xkcd:red orange         | #fd3c06 | $\fcolorbox{gray}{#fd3c06}{\phantom{x}}$ |
|        xkcd:dark magenta        | #960056 | $\fcolorbox{gray}{#960056}{\phantom{x}}$ |
|          xkcd:greenish          | #40a368 | $\fcolorbox{gray}{#40a368}{\phantom{x}}$ |
|         xkcd:ocean blue         | #03719c | $\fcolorbox{gray}{#03719c}{\phantom{x}}$ |
|           xkcd:coral            | #fc5a50 | $\fcolorbox{gray}{#fc5a50}{\phantom{x}}$ |
|           xkcd:cream            | #ffffc2 | $\fcolorbox{gray}{#ffffc2}{\phantom{x}}$ |
|       xkcd:reddish brown        | #7f2b0a | $\fcolorbox{gray}{#7f2b0a}{\phantom{x}}$ |
|        xkcd:burnt sienna        | #b04e0f | $\fcolorbox{gray}{#b04e0f}{\phantom{x}}$ |
|           xkcd:brick            | #a03623 | $\fcolorbox{gray}{#a03623}{\phantom{x}}$ |
|            xkcd:sage            | #87ae73 | $\fcolorbox{gray}{#87ae73}{\phantom{x}}$ |
|         xkcd:grey green         | #789b73 | $\fcolorbox{gray}{#789b73}{\phantom{x}}$ |
|           xkcd:white            | #ffffff | $\fcolorbox{gray}{#ffffff}{\phantom{x}}$ |
|      xkcd:robin's egg blue      | #98eff9 | $\fcolorbox{gray}{#98eff9}{\phantom{x}}$ |
|         xkcd:moss green         | #658b38 | $\fcolorbox{gray}{#658b38}{\phantom{x}}$ |
|         xkcd:steel blue         | #5a7d9a | $\fcolorbox{gray}{#5a7d9a}{\phantom{x}}$ |
|          xkcd:eggplant          | #380835 | $\fcolorbox{gray}{#380835}{\phantom{x}}$ |
|        xkcd:light yellow        | #fffe7a | $\fcolorbox{gray}{#fffe7a}{\phantom{x}}$ |
|         xkcd:leaf green         | #5ca904 | $\fcolorbox{gray}{#5ca904}{\phantom{x}}$ |
|         xkcd:light grey         | #d8dcd6 | $\fcolorbox{gray}{#d8dcd6}{\phantom{x}}$ |
|            xkcd:puke            | #a5a502 | $\fcolorbox{gray}{#a5a502}{\phantom{x}}$ |
|       xkcd:pinkish purple       | #d648d7 | $\fcolorbox{gray}{#d648d7}{\phantom{x}}$ |
|          xkcd:sea blue          | #047495 | $\fcolorbox{gray}{#047495}{\phantom{x}}$ |
|        xkcd:pale purple         | #b790d4 | $\fcolorbox{gray}{#b790d4}{\phantom{x}}$ |
|         xkcd:slate blue         | #5b7c99 | $\fcolorbox{gray}{#5b7c99}{\phantom{x}}$ |
|         xkcd:blue grey          | #607c8e | $\fcolorbox{gray}{#607c8e}{\phantom{x}}$ |
|        xkcd:hunter green        | #0b4008 | $\fcolorbox{gray}{#0b4008}{\phantom{x}}$ |
|          xkcd:fuchsia           | #ed0dd9 | $\fcolorbox{gray}{#ed0dd9}{\phantom{x}}$ |
|          xkcd:crimson           | #8c000f | $\fcolorbox{gray}{#8c000f}{\phantom{x}}$ |
|        xkcd:pale yellow         | #ffff84 | $\fcolorbox{gray}{#ffff84}{\phantom{x}}$ |
|           xkcd:ochre            | #bf9005 | $\fcolorbox{gray}{#bf9005}{\phantom{x}}$ |
|       xkcd:mustard yellow       | #d2bd0a | $\fcolorbox{gray}{#d2bd0a}{\phantom{x}}$ |
|         xkcd:light red          | #ff474c | $\fcolorbox{gray}{#ff474c}{\phantom{x}}$ |
|          xkcd:cerulean          | #0485d1 | $\fcolorbox{gray}{#0485d1}{\phantom{x}}$ |
|         xkcd:pale pink          | #ffcfdc | $\fcolorbox{gray}{#ffcfdc}{\phantom{x}}$ |
|         xkcd:deep blue          | #040273 | $\fcolorbox{gray}{#040273}{\phantom{x}}$ |
|            xkcd:rust            | #a83c09 | $\fcolorbox{gray}{#a83c09}{\phantom{x}}$ |
|         xkcd:light teal         | #90e4c1 | $\fcolorbox{gray}{#90e4c1}{\phantom{x}}$ |
|           xkcd:slate            | #516572 | $\fcolorbox{gray}{#516572}{\phantom{x}}$ |
|         xkcd:goldenrod          | #fac205 | $\fcolorbox{gray}{#fac205}{\phantom{x}}$ |
|        xkcd:dark yellow         | #d5b60a | $\fcolorbox{gray}{#d5b60a}{\phantom{x}}$ |
|         xkcd:dark grey          | #363737 | $\fcolorbox{gray}{#363737}{\phantom{x}}$ |
|         xkcd:army green         | #4b5d16 | $\fcolorbox{gray}{#4b5d16}{\phantom{x}}$ |
|         xkcd:grey blue          | #6b8ba4 | $\fcolorbox{gray}{#6b8ba4}{\phantom{x}}$ |
|          xkcd:seafoam           | #80f9ad | $\fcolorbox{gray}{#80f9ad}{\phantom{x}}$ |
|            xkcd:puce            | #a57e52 | $\fcolorbox{gray}{#a57e52}{\phantom{x}}$ |
|        xkcd:spring green        | #a9f971 | $\fcolorbox{gray}{#a9f971}{\phantom{x}}$ |
|        xkcd:dark orange         | #c65102 | $\fcolorbox{gray}{#c65102}{\phantom{x}}$ |
|            xkcd:sand            | #e2ca76 | $\fcolorbox{gray}{#e2ca76}{\phantom{x}}$ |
|        xkcd:pastel green        | #b0ff9d | $\fcolorbox{gray}{#b0ff9d}{\phantom{x}}$ |
|            xkcd:mint            | #9ffeb0 | $\fcolorbox{gray}{#9ffeb0}{\phantom{x}}$ |
|        xkcd:light orange        | #fdaa48 | $\fcolorbox{gray}{#fdaa48}{\phantom{x}}$ |
|        xkcd:bright pink         | #fe01b1 | $\fcolorbox{gray}{#fe01b1}{\phantom{x}}$ |
|         xkcd:chartreuse         | #c1f80a | $\fcolorbox{gray}{#c1f80a}{\phantom{x}}$ |
|        xkcd:deep purple         | #36013f | $\fcolorbox{gray}{#36013f}{\phantom{x}}$ |
|         xkcd:dark brown         | #341c02 | $\fcolorbox{gray}{#341c02}{\phantom{x}}$ |
|           xkcd:taupe            | #b9a281 | $\fcolorbox{gray}{#b9a281}{\phantom{x}}$ |
|         xkcd:pea green          | #8eab12 | $\fcolorbox{gray}{#8eab12}{\phantom{x}}$ |
|         xkcd:puke green         | #9aae07 | $\fcolorbox{gray}{#9aae07}{\phantom{x}}$ |
|        xkcd:kelly green         | #02ab2e | $\fcolorbox{gray}{#02ab2e}{\phantom{x}}$ |
|       xkcd:seafoam green        | #7af9ab | $\fcolorbox{gray}{#7af9ab}{\phantom{x}}$ |
|         xkcd:blue green         | #137e6d | $\fcolorbox{gray}{#137e6d}{\phantom{x}}$ |
|           xkcd:khaki            | #aaa662 | $\fcolorbox{gray}{#aaa662}{\phantom{x}}$ |
|          xkcd:burgundy          | #610023 | $\fcolorbox{gray}{#610023}{\phantom{x}}$ |
|         xkcd:dark teal          | #014d4e | $\fcolorbox{gray}{#014d4e}{\phantom{x}}$ |
|         xkcd:brick red          | #8f1402 | $\fcolorbox{gray}{#8f1402}{\phantom{x}}$ |
|        xkcd:royal purple        | #4b006e | $\fcolorbox{gray}{#4b006e}{\phantom{x}}$ |
|            xkcd:plum            | #580f41 | $\fcolorbox{gray}{#580f41}{\phantom{x}}$ |
|         xkcd:mint green         | #8fff9f | $\fcolorbox{gray}{#8fff9f}{\phantom{x}}$ |
|            xkcd:gold            | #dbb40c | $\fcolorbox{gray}{#dbb40c}{\phantom{x}}$ |
|         xkcd:baby blue          | #a2cffe | $\fcolorbox{gray}{#a2cffe}{\phantom{x}}$ |
|        xkcd:yellow green        | #c0fb2d | $\fcolorbox{gray}{#c0fb2d}{\phantom{x}}$ |
|       xkcd:bright purple        | #be03fd | $\fcolorbox{gray}{#be03fd}{\phantom{x}}$ |
|          xkcd:dark red          | #840000 | $\fcolorbox{gray}{#840000}{\phantom{x}}$ |
|         xkcd:pale blue          | #d0fefe | $\fcolorbox{gray}{#d0fefe}{\phantom{x}}$ |
|        xkcd:grass green         | #3f9b0b | $\fcolorbox{gray}{#3f9b0b}{\phantom{x}}$ |
|            xkcd:navy            | #01153e | $\fcolorbox{gray}{#01153e}{\phantom{x}}$ |
|         xkcd:aquamarine         | #04d8b2 | $\fcolorbox{gray}{#04d8b2}{\phantom{x}}$ |
|        xkcd:burnt orange        | #c04e01 | $\fcolorbox{gray}{#c04e01}{\phantom{x}}$ |
|         xkcd:neon green         | #0cff0c | $\fcolorbox{gray}{#0cff0c}{\phantom{x}}$ |
|        xkcd:bright blue         | #0165fc | $\fcolorbox{gray}{#0165fc}{\phantom{x}}$ |
|            xkcd:rose            | #cf6275 | $\fcolorbox{gray}{#cf6275}{\phantom{x}}$ |
|         xkcd:light pink         | #ffd1df | $\fcolorbox{gray}{#ffd1df}{\phantom{x}}$ |
|          xkcd:mustard           | #ceb301 | $\fcolorbox{gray}{#ceb301}{\phantom{x}}$ |
|           xkcd:indigo           | #380282 | $\fcolorbox{gray}{#380282}{\phantom{x}}$ |
|            xkcd:lime            | #aaff32 | $\fcolorbox{gray}{#aaff32}{\phantom{x}}$ |
|         xkcd:sea green          | #53fca1 | $\fcolorbox{gray}{#53fca1}{\phantom{x}}$ |
|         xkcd:periwinkle         | #8e82fe | $\fcolorbox{gray}{#8e82fe}{\phantom{x}}$ |
|         xkcd:dark pink          | #cb416b | $\fcolorbox{gray}{#cb416b}{\phantom{x}}$ |
|        xkcd:olive green         | #677a04 | $\fcolorbox{gray}{#677a04}{\phantom{x}}$ |
|           xkcd:peach            | #ffb07c | $\fcolorbox{gray}{#ffb07c}{\phantom{x}}$ |
|         xkcd:pale green         | #c7fdb5 | $\fcolorbox{gray}{#c7fdb5}{\phantom{x}}$ |
|        xkcd:light brown         | #ad8150 | $\fcolorbox{gray}{#ad8150}{\phantom{x}}$ |
|          xkcd:hot pink          | #ff028d | $\fcolorbox{gray}{#ff028d}{\phantom{x}}$ |
|           xkcd:black            | #000000 | $\fcolorbox{gray}{#000000}{\phantom{x}}$ |
|           xkcd:lilac            | #cea2fd | $\fcolorbox{gray}{#cea2fd}{\phantom{x}}$ |
|         xkcd:navy blue          | #001146 | $\fcolorbox{gray}{#001146}{\phantom{x}}$ |
|         xkcd:royal blue         | #0504aa | $\fcolorbox{gray}{#0504aa}{\phantom{x}}$ |
|           xkcd:beige            | #e6daa6 | $\fcolorbox{gray}{#e6daa6}{\phantom{x}}$ |
|           xkcd:salmon           | #ff796c | $\fcolorbox{gray}{#ff796c}{\phantom{x}}$ |
|           xkcd:olive            | #6e750e | $\fcolorbox{gray}{#6e750e}{\phantom{x}}$ |
|           xkcd:maroon           | #650021 | $\fcolorbox{gray}{#650021}{\phantom{x}}$ |
|        xkcd:bright green        | #01ff07 | $\fcolorbox{gray}{#01ff07}{\phantom{x}}$ |
|        xkcd:dark purple         | #35063e | $\fcolorbox{gray}{#35063e}{\phantom{x}}$ |
|           xkcd:mauve            | #ae7181 | $\fcolorbox{gray}{#ae7181}{\phantom{x}}$ |
|        xkcd:forest green        | #06470c | $\fcolorbox{gray}{#06470c}{\phantom{x}}$ |
|            xkcd:aqua            | #13eac9 | $\fcolorbox{gray}{#13eac9}{\phantom{x}}$ |
|            xkcd:cyan            | #00ffff | $\fcolorbox{gray}{#00ffff}{\phantom{x}}$ |
|            xkcd:tan             | #d1b26f | $\fcolorbox{gray}{#d1b26f}{\phantom{x}}$ |
|         xkcd:dark blue          | #00035b | $\fcolorbox{gray}{#00035b}{\phantom{x}}$ |
|          xkcd:lavender          | #c79fef | $\fcolorbox{gray}{#c79fef}{\phantom{x}}$ |
|         xkcd:turquoise          | #06c2ac | $\fcolorbox{gray}{#06c2ac}{\phantom{x}}$ |
|         xkcd:dark green         | #033500 | $\fcolorbox{gray}{#033500}{\phantom{x}}$ |
|           xkcd:violet           | #9a0eea | $\fcolorbox{gray}{#9a0eea}{\phantom{x}}$ |
|        xkcd:light purple        | #bf77f6 | $\fcolorbox{gray}{#bf77f6}{\phantom{x}}$ |
|         xkcd:lime green         | #89fe05 | $\fcolorbox{gray}{#89fe05}{\phantom{x}}$ |
|            xkcd:grey            | #929591 | $\fcolorbox{gray}{#929591}{\phantom{x}}$ |
|          xkcd:sky blue          | #75bbfd | $\fcolorbox{gray}{#75bbfd}{\phantom{x}}$ |
|           xkcd:yellow           | #ffff14 | $\fcolorbox{gray}{#ffff14}{\phantom{x}}$ |
|          xkcd:magenta           | #c20078 | $\fcolorbox{gray}{#c20078}{\phantom{x}}$ |
|        xkcd:light green         | #96f97b | $\fcolorbox{gray}{#96f97b}{\phantom{x}}$ |
|           xkcd:orange           | #f97306 | $\fcolorbox{gray}{#f97306}{\phantom{x}}$ |
|            xkcd:teal            | #029386 | $\fcolorbox{gray}{#029386}{\phantom{x}}$ |
|         xkcd:light blue         | #95d0fc | $\fcolorbox{gray}{#95d0fc}{\phantom{x}}$ |
|            xkcd:red             | #e50000 | $\fcolorbox{gray}{#e50000}{\phantom{x}}$ |
|           xkcd:brown            | #653700 | $\fcolorbox{gray}{#653700}{\phantom{x}}$ |
|            xkcd:pink            | #ff81c0 | $\fcolorbox{gray}{#ff81c0}{\phantom{x}}$ |
|            xkcd:blue            | #0343df | $\fcolorbox{gray}{#0343df}{\phantom{x}}$ |
|           xkcd:green            | #15b01a | $\fcolorbox{gray}{#15b01a}{\phantom{x}}$ |
|           xkcd:purple           | #7e1e9c | $\fcolorbox{gray}{#7e1e9c}{\phantom{x}}$ |
