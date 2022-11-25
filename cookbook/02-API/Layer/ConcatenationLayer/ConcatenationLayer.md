# Concatenation Layer

+ Simple example
+ axis

---

## Simple example

+ Refer to SimpleExample.py

+ Shape of input tensor 0: (1,3,4,5)
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

+ By default, concatenate on the highest dimension beside Batch dimension, shape of output tensor 0: (1,6,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right] \\
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## axis

+ Refer to Axis.py, set the axis of concatenation

+ default value: min(len(inputT0.shape), 3)

+ set **axis=0** to concatenate on the highest dimension, shape of output tensor 0: (2,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  1. &  2. &  3. &  4. \\
             2. &  6. &  7. &  8. &  9. \\
            1.  & 11. & 12. & 13. & 14. \\
            2.  & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.  & 21. & 22. & 23. & 24. \\
            2.  & 26. & 27. & 28. & 29. \\
            3.  & 31. & 32. & 33. & 34. \\
            4.  & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.  & 41. & 42. & 43. & 44. \\
            2.  & 46. & 47. & 48. & 49. \\
            3.  & 51. & 52. & 53. & 54. \\
            4.  & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  1. &  2. &  3. &  4. \\
             2. &  6. &  7. &  8. &  9. \\
            1.  & 11. & 12. & 13. & 14. \\
            2.  & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.  & 21. & 22. & 23. & 24. \\
            2.  & 26. & 27. & 28. & 29. \\
            3.  & 31. & 32. & 33. & 34. \\
            4.  & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.  & 41. & 42. & 43. & 44. \\
            2.  & 46. & 47. & 48. & 49. \\
            3.  & 51. & 52. & 53. & 54. \\
            4.  & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ set **axis=1** to concatenate on the second highest dimension, shape of output tensor 0: (1,6,4,5), which is the same as default example.

+ set **axis=2** to concatenate on the thrid highest dimension, shape of output tensor 0: (1,3,8,5).
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  2. &  3. &  4. \\
         2. &  6. &  7. &  8. &  9. \\
        1.  & 11. & 12. & 13. & 14. \\
        2.  & 16. & 17. & 18. & 19.\\
         3. &  1. &  2. &  3. &  4. \\
         4. &  6. &  7. &  8. &  9. \\
        3.  & 11. & 12. & 13. & 14. \\
        4.  & 16. & 17. & 18. & 19.\\
    \end{matrix}\right]
    \left[\begin{matrix}
        1.  & 21. & 22. & 23. & 24. \\
        2.  & 26. & 27. & 28. & 29. \\
        3.  & 31. & 32. & 33. & 34. \\
        4.  & 36. & 37. & 38. & 39. \\
        5.  & 21. & 22. & 23. & 24. \\
        6.  & 26. & 27. & 28. & 29. \\
        7.  & 31. & 32. & 33. & 34. \\
        8.  & 36. & 37. & 38. & 39. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        1.  & 41. & 42. & 43. & 44. \\
        2.  & 46. & 47. & 48. & 49. \\
        3.  & 51. & 52. & 53. & 54. \\
        4.  & 56. & 57. & 58. & 59. \\
        5.  & 41. & 42. & 43. & 44. \\
        6.  & 46. & 47. & 48. & 49. \\
        7.  & 51. & 52. & 53. & 54. \\
        8.  & 56. & 57. & 58. & 59. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ set **axis=3** to concatenate on the fourth highest dimension, shape of output tensor 0: (1,3,4,10).
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  2. &  3. &  4. &  0. &  1. &  2. &  3. &  4. \\
         2. &  6. &  7. &  8. &  9. &  5. &  6. &  7. &  8. &  9. \\
        1.  & 11. & 12. & 13. & 14. & 10. & 11. & 12. & 13. & 14. \\
        2.  & 16. & 17. & 18. & 19. & 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right] \\
    \left[\begin{matrix}
        1.  & 21. & 22. & 23. & 24. & 20. & 21. & 22. & 23. & 24. \\
        2.  & 26. & 27. & 28. & 29. & 25. & 26. & 27. & 28. & 29. \\
        3.  & 31. & 32. & 33. & 34. & 30. & 31. & 32. & 33. & 34. \\
        4.  & 36. & 37. & 38. & 39. & 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right] \\
    \left[\begin{matrix}
        1.  & 41. & 42. & 43. & 44. & 40. & 41. & 42. & 43. & 44. \\
        2.  & 46. & 47. & 48. & 49. & 45. & 46. & 47. & 48. & 49. \\
        3.  & 51. & 52. & 53. & 54. & 50. & 51. & 52. & 53. & 54. \\
        4.  & 56. & 57. & 58. & 59. & 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$
