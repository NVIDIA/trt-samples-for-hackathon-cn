### TensorRT 7
```
Input 0: (1, 3, 4, 5) 
 [[[[ 0.  1.  2.  3.  4.]
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
   [55. 56. 57. 58. 59.]]]]
Output 0: (1, 3, 4, 4) 
 [[[[ 10.  10.  10.  10.]
   [ 35.  35.  35.  35.]
   [ 60.  60.  60.  60.]
   [ 85.  85.  85.  85.]]

  [[110. 110. 110. 110.]
   [135. 135. 135. 135.]
   [160. 160. 160. 160.]
   [185. 185. 185. 185.]]

  [[210. 210. 210. 210.]
   [235. 235. 235. 235.]
   [260. 260. 260. 260.]
   [285. 285. 285. 285.]]]]
```

### TensorRT 8
``````
Traceback (most recent call last):
  File "main.py", line 37, in <module>
    matrixMultiplyLayer = network.add_matrix_multiply_deprecated(inputT0, True, constantLayer.get_output(0), True)
AttributeError: 'tensorrt.tensorrt.INetworkDefinition' object has no attribute 'add_matrix_multiply_deprecated'
``````