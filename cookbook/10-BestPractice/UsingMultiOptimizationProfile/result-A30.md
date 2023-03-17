# usign one Optimization Profile (min=1，opt=510，max=512) 

```shell
Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (1, 1) tensor0
Bind[ 1]:o[ 0]-> DataType.FLOAT (-1,) (1,) tensor8
+---- BatchSize= 1: 0.5007ms

Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (4, 1) tensor0
Bind[ 1]:o[ 0]-> DataType.FLOAT (-1,) (4,) tensor8
+---- BatchSize= 4: 0.4999ms

Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (510, 1) tensor0
Bind[ 1]:o[ 0]-> DataType.FLOAT (-1,) (510,) tensor8
+---- BatchSize=510: 0.5307ms

Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (512, 1) tensor0
Bind[ 1]:o[ 0]-> DataType.FLOAT (-1,) (512,) tensor8
+---- BatchSize=512: 0.5308ms
```

# using two Optimization Profile (min=1，opt=4，max=4) + (min=510，opt=510，max=512) 

```shell
Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (1, 1) tensor0
Bind[ 1]:i[ 1]-> DataType.FLOAT (-1,) (1,) tensor8
Bind[ 2]:o[ 0]-> DataType.FLOAT (-1, 1) (-1, 1) tensor0 [profile 1]
Bind[ 3]:o[ 1]-> DataType.FLOAT (-1,) (-1,) tensor8 [profile 1]
+---- BatchSize= 1: 0.2409ms

Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (4, 1) tensor0
Bind[ 1]:i[ 1]-> DataType.FLOAT (-1,) (4,) tensor8
Bind[ 2]:o[ 0]-> DataType.FLOAT (-1, 1) (-1, 1) tensor0 [profile 1]
Bind[ 3]:o[ 1]-> DataType.FLOAT (-1,) (-1,) tensor8 [profile 1]
+---- BatchSize= 4: 0.2668ms

Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (-1, 1) tensor0
Bind[ 1]:i[ 1]-> DataType.FLOAT (-1,) (-1,) tensor8
Bind[ 2]:o[ 0]-> DataType.FLOAT (-1, 1) (510, 1) tensor0 [profile 1]
Bind[ 3]:o[ 1]-> DataType.FLOAT (-1,) (510,) tensor8 [profile 1]
+---- BatchSize=510: 0.5288ms

Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, 1) (-1, 1) tensor0
Bind[ 1]:i[ 1]-> DataType.FLOAT (-1,) (-1,) tensor8
Bind[ 2]:o[ 0]-> DataType.FLOAT (-1, 1) (512, 1) tensor0 [profile 1]
Bind[ 3]:o[ 1]-> DataType.FLOAT (-1,) (512,) tensor8 [profile 1]
+---- BatchSize=512: 0.5272ms
```
