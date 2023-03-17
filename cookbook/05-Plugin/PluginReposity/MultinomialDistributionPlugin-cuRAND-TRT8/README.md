# RandomPlugin
+ 用给定的概率分布列采样非负随机整数，并计算该取值的熵
+ Input tensor:
    - [0]: (nBatchSize, nRow, nCol)     int32
+ Input parameter:
    - 无
+ Output tensor:
    - [0]: (nBatchSize, nRow)           int32,
    - [1]: (nBatchSize, nRow)           float32/float16
+ Steps to run：`make test`
+ Output for reference: ./result.log

# Result:
+ RandomByCub
    - distribution30
```
0 -> 0, 3.0332
1 -> 6, 2.4235
2 -> 3, 2.7532
3 -> 1, 2.9434
4 -> 21, 2.8643
5 -> 20, 4.6734
6 -> 19, 4.0522
7 -> 21, 3.0050
8 -> 29, 2.5096
9 -> 15, 3.1641
10 -> 8, 3.5645
11 -> 12, 3.0052
12 -> 17, 2.7333
13 -> 28, 2.8564
14 -> 1, 3.6896
15 -> 25, 3.0315
```
    - distribution192
```
  0 ->   0	-0.9057
  1 ->   0	-1.0654
  2 ->   1	-1.5507
  3 ->   0	-0.9792
  4 ->  26	-1.3117
  5 ->  23	0.7938
  6 ->   0	-0.8423
  7 ->  17	4.0732
  8 ->  46	-3.5481
  9 ->   0	-3.3777
 10 ->   0	-0.0438
 11 ->   4	0.4278
 12 ->   1	1.0881
 13 ->  51	3.4246
 14 ->   0	0.1180
 15 ->  17	0.2768
```

+ V1.0
```
python3 ./testRandomPlugin.py
test: nRow=320,nCol=30
Succeeded building engine!
outputH0
(320,) mean=14.12,var=77.32,max=29,min=0
outputH1
(320,) mean=3.40
test: nRow=320,nCol=9
Succeeded building engine!
outputH0
(320,) mean=3.89,var=6.93,max=8,min=0
outputH1
(320,) mean=2.20
test: nRow=320,nCol=4
Succeeded building engine!
outputH0
(320,) mean=1.47,var=1.32,max=3,min=0
outputH1
(320,) mean=1.39
test: nRow=320,nCol=192
Succeeded building engine!
outputH0
(320,) mean=93.08,var=3175.62,max=191,min=0
outputH1
(320,) mean=5.26
test finish!
```

+ V1.9
```
python3 ./testRandomPlugin.py
test: nRow=320,nCol=30
Succeeded building engine!
>320,30,0
outputH0
(320, 1) mean=13.88,var=72.96,max=29,min=0
outputH1
(320, 1) mean=3.40
test: nRow=320,nCol=9
Succeeded building engine!
>320,9,0
outputH0
(320, 1) mean=4.03,var=5.93,max=8,min=0
outputH1
(320, 1) mean=2.20
test: nRow=320,nCol=4
Succeeded building engine!
>320,4,0
outputH0
(320, 1) mean=1.48,var=1.24,max=3,min=0
outputH1
(320, 1) mean=1.39
test: nRow=320,nCol=192
Succeeded building engine!
>320,192,0
outputH0
(320, 1) mean=101.03,var=3162.55,max=191,min=0
outputH1
(320, 1) mean=5.26
test finish!
```
