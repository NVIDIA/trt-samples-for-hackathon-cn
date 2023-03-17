#

## Introduction

+ We test the code in four scenarios.

  1. [32,1] -> [32,256] -> [32,2048] -> [32,256] -> [32,2048] -> ... -> [32,256]
  2. [31,1] -> [31,256] -> [31,2048] -> [31,256] -> [31,2048] -> ... -> [31,256]
  3. [32,1] -> [32,255] -> [32,2048] -> [32,255] -> [32,2048] -> ... -> [32,255]
  4. [32,1] -> [32,256] -> [32,2047] -> [32,256] -> [32,2047] -> ... -> [32,256]

## Result

+ When using python script, the performance of scenario 1 and scenario 2 is close to that of scenario 3 and scenario 4.
