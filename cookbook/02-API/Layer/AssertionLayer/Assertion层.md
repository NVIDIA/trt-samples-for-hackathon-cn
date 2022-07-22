# Assertion 层
+ 初始示例代码
+ message
+ 运行期检查

---
### 初始示例代码
+ 见 SimpleUsage.py，相当于构建期检查

+ 输入张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,)，构建期检查 inputT0.shape[3] 是否为 5，根据上下文该检查肯定是通过的
$$
\left[\begin{matrix}
    1
\end{matrix}\right]
$$

+ 将代码网络部分改为 `SimpleUsage2.py` 的形式，即构建期检查 inputT0.shape[3] 是否为 4，根据上下文该检查肯定不通过，构建期会收到报错信息：
```
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::581] Error Code 4: Internal Error (IAssertionLayer (Unnamed Layer* 5) [Assertion]: condition[0] is false: 0. inputT0.shape[3] is not 4!)
```

---
### message
+ 见 Message.py，修改 assert 层的警告信息。在初始示例代码中 _HA 定义的下面添加一行 `_HA.message = "edited message!"`
    
+ 输入输出张量同 SimpleUsage.py，但是改变了报错信息的内容

---
### 运行期检查
+ 见 RuntimeCheck.py，执行运行期检查。检查两个输入张量的第 1 维度长度是否相等，即 ` inputT0.shape[1] == inputT1.shape[1]` 是否成立

+ 输入张量同 SimpleUsage.py

+ 这里当输入张量 1 形状 (1, 3) 时检查通过，输出张量 0 形状（1,）
$$
\left[\begin{matrix}
    1
\end{matrix}\right]
$$

+ 当输入张量 1 形状 (1, 4) 时检查不通过，报错信息如下：
```
[TRT] [E] 7: [shapeMachine.cpp::execute::702] Error Code 7: Internal Error (IAssertionLayer (Unnamed Layer* 6) [Assertion]: condition[0] is false: (EQUAL (# 1 (SHAPE inputT1)) (# 1 (SHAPE inputT0))). inputT0.shape[:2] != inputT1.shape[:2]
condition '==' violated. 3 != 4
Instruction: CHECK_EQUAL 3 4
)
```