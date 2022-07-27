make clean
make[1]: Entering directory '/work/gitlab/tensorrt-cookbook-in-chinese/01-SimpleDemo/TensorRT8'
rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan
make[1]: Leaving directory '/work/gitlab/tensorrt-cookbook-in-chinese/01-SimpleDemo/TensorRT8'
make
make[1]: Entering directory '/work/gitlab/tensorrt-cookbook-in-chinese/01-SimpleDemo/TensorRT8'
/usr/local/cuda/bin/nvcc -w -std=c++14 -O3 -UDEBUG -Xcompiler -fPIC -use_fast_math -I. -I/usr/local/cuda/include -I/usr/lib/x86_64-linux-gnu/include -M -MT main.o -o main.d main.cpp
/usr/local/cuda/bin/nvcc -w -std=c++14 -O3 -UDEBUG -Xcompiler -fPIC -use_fast_math -I. -I/usr/local/cuda/include -I/usr/lib/x86_64-linux-gnu/include -Xcompiler -fPIC -o main.o -c main.cpp
/usr/local/cuda/bin/nvcc -w -std=c++14 -O3 -UDEBUG -Xcompiler -fPIC -use_fast_math -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu/lib -lnvinfer -o main.exe main.o
rm main.o
make[1]: Leaving directory '/work/gitlab/tensorrt-cookbook-in-chinese/01-SimpleDemo/TensorRT8'
python3 ./main-cudart.py
Succeeded getting serialized engine!
Succeeded saving .plan file!
Succeeded building engine!
Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, -1, -1) (3, 4, 5) inputT0
Bind[ 1]:o[ 0]-> DataType.FLOAT (-1, -1, -1) (3, 4, 5) (Unnamed Layer* 0) [Identity]_output
inputT0
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
(Unnamed Layer* 0) [Identity]_output
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
Succeeded getting serialized engine!
Succeeded building engine!
Bind[ 0]:i[ 0]-> DataType.FLOAT (-1, -1, -1) (3, 4, 5) inputT0
Bind[ 1]:o[ 0]-> DataType.FLOAT (-1, -1, -1) (3, 4, 5) (Unnamed Layer* 0) [Identity]_output
inputT0
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
(Unnamed Layer* 0) [Identity]_output
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
rm ./*.plan
././main.exe
Succeeded getting serialized engine!
Succeeded building engine!
Succeeded saving .plan file!
inputT0: (3, 4, 5)
 0.0  1.0  2.0  3.0  4.0 
 4.0  5.0  6.0  7.0  8.0 
 8.0  9.0 10.0 11.0 12.0 
12.0 13.0 14.0 15.0 16.0 

12.0 13.0 14.0 15.0 16.0 
16.0 17.0 18.0 19.0 20.0 
20.0 21.0 22.0 23.0 24.0 
24.0 25.0 26.0 27.0 28.0 

24.0 25.0 26.0 27.0 28.0 
28.0 29.0 30.0 31.0 32.0 
32.0 33.0 34.0 35.0 36.0 
36.0 37.0 38.0 39.0 40.0 

(Unnamed Layer* 0) [Identity]_output: (3, 4, 5)
 0.0  1.0  2.0  3.0  4.0 
 4.0  5.0  6.0  7.0  8.0 
 8.0  9.0 10.0 11.0 12.0 
12.0 13.0 14.0 15.0 16.0 

12.0 13.0 14.0 15.0 16.0 
16.0 17.0 18.0 19.0 20.0 
20.0 21.0 22.0 23.0 24.0 
24.0 25.0 26.0 27.0 28.0 

24.0 25.0 26.0 27.0 28.0 
28.0 29.0 30.0 31.0 32.0 
32.0 33.0 34.0 35.0 36.0 
36.0 37.0 38.0 39.0 40.0 

Succeeded getting serialized engine!
Succeeded building engine!
inputT0: (3, 4, 5)
 0.0  1.0  2.0  3.0  4.0 
 4.0  5.0  6.0  7.0  8.0 
 8.0  9.0 10.0 11.0 12.0 
12.0 13.0 14.0 15.0 16.0 

12.0 13.0 14.0 15.0 16.0 
16.0 17.0 18.0 19.0 20.0 
20.0 21.0 22.0 23.0 24.0 
24.0 25.0 26.0 27.0 28.0 

24.0 25.0 26.0 27.0 28.0 
28.0 29.0 30.0 31.0 32.0 
32.0 33.0 34.0 35.0 36.0 
36.0 37.0 38.0 39.0 40.0 

(Unnamed Layer* 0) [Identity]_output: (3, 4, 5)
 0.0  1.0  2.0  3.0  4.0 
 4.0  5.0  6.0  7.0  8.0 
 8.0  9.0 10.0 11.0 12.0 
12.0 13.0 14.0 15.0 16.0 

12.0 13.0 14.0 15.0 16.0 
16.0 17.0 18.0 19.0 20.0 
20.0 21.0 22.0 23.0 24.0 
24.0 25.0 26.0 27.0 28.0 

24.0 25.0 26.0 27.0 28.0 
28.0 29.0 30.0 31.0 32.0 
32.0 33.0 34.0 35.0 36.0 
36.0 37.0 38.0 39.0 40.0 

