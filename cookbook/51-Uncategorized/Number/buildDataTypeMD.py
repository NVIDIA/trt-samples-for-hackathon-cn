#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse


def buildMD(nSign, nExponent, nMantissa, nBias, bIEEE754=True):  # Bias is useless now

    typeName = "S%dE%dM%dB%d" % (nSign, nExponent, nMantissa, nBias)
    if typeName == "S1E11M52B0":
        typeName += "(FP64)"
    elif typeName == "S1E8M23B0":
        typeName += "(FP32)"
    elif typeName == "S1E8M10B0":
        typeName += "(TF32)"
    elif typeName == "S1E5M10B0":
        typeName += "(FP16)"
    elif typeName == "S1E8M7B0":
        typeName += "(BF16)"
    elif typeName == "S1E5M2B0":
        typeName += "(FP8e5m2)"
    elif typeName == "S1E4M3B0":
        typeName += "(FP8e4m3)"

    print("Build %s" % typeName)

    if nExponent < 2:
        print("nExponent should be equal or greater than 2!")

    ss = ""

    def numberToString(x):
        return ("%6.4e" % x).split("e")

    with open(typeName + ".md", "w") as f:

        ss += "# %s - %s\n" % (typeName, ("" if bIEEE754 else "not ") + "IEEE754")
        ss += "\n"
        ss += "+ SignBit ($s$): %d\n" % nSign
        ss += "+ Exponent ($k$): %d\n" % nExponent
        ss += "+ Mantissa ($n$): %d\n" % nMantissa
        ss += "+ Bias ($b$): %d\n" % nBias
        ss += "\n"
        ss += "+ Special value\n"
        ss += "| Mantissa   | all 0             | not all 0         |\n"
        ss += "| :-:        | :-:               | :-:               |\n"
        ss += "| $e = %s_2$ | Signed Zero       | Subnormal Value   |\n" % ("0" * nExponent)
        ss += "| $e = %s_2$ | Signed Infinity   |       NaN         |\n" % ("1" * nExponent)
        ss += "\n"
        ss += "+ Normal value ($%s1_2 \le e_2 \le %s0_2$)\n" % ("0" * (nExponent - 1), "1" * (nExponent - 1))
        ss += "$$\n"
        ss += "\\begin{equation}\n"
        ss += "\\begin{aligned}\n"
        ss += "E &= e_{10} - \left( 2^{k-1} - 1 \\right) \\\\\n"
        ss += "M &= f_{10} \cdot 2^{-n} \\\\\n"
        ss += "value &= \left(-1\\right)^{s}2^{E}\left(1+M\\right)\n"
        ss += "\end{aligned}\n"
        ss += "\end{equation}\n"
        ss += "$$\n"
        ss += "\n"
        ss += "+ Subnormal value ($e_2 = %s_2$)\n" % ("0" * nExponent)
        ss += "$$\n"
        ss += "\\begin{equation}\n"
        ss += "\\begin{aligned}\n"
        ss += "E &= 2 - 2^{k-1} = %d \\\\\n" % (2 - 2 ** (nExponent - 1))
        ss += "M &= f_{10} \cdot 2^{-n} \\\\\n"
        ss += "value &= \left(-1\\right)^{s}2^{E}M\n"
        ss += "\end{aligned}\n"
        ss += "\end{equation}\n"
        ss += "$$\n"
        ss += "\n"
        ss += "+ Examples\n"

        # yapf:disable
        ss += "| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponent}\color{#0000FF}{Mantissa}$)  | value                 |        comment        |\n"
        ss += "| :-:                                                                                | :-:                   | :-:                   |\n"
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{%s}$                         | $+0$                  |                       |\n" % ("0" * nExponent,        "0" * nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{%s1}$                        | $%s\\times10^{%s}$    |   Minimum subnormal   |\n" % ("0" * nExponent,        "0" * (nMantissa - 1),  *numberToString(        2 ** (2 - 2 ** (nExponent - 1)) * (0 + (1) / (2 ** nMantissa))))
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{%s}$                         | $%s\\times10^{%s}$    |   Maximum subnormal   |\n" % ("0" * nExponent,        "1" * nMantissa,        *numberToString(        2 ** (2 - 2 ** (nExponent - 1)) * (0 + (2 ** nMantissa - 1) / (2 ** nMantissa))))
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s1}\color{#0000FF}{%s}$                        | $%s\\times10^{%s}$    |    Minimum normal     |\n" % ("0" * (nExponent - 1),  "0" * nMantissa,        *numberToString(        2 ** (2 - 2 ** (nExponent - 1)) * (1 + (0) / (2 ** nMantissa))))
        if nExponent >= 3:
            ss += "| $\color{#FF0000}{0}\color{#007F00}{0%s0}\color{#0000FF}{%s}$                   |  $1 - 2^{-%d}$        |  largest number < 1   |\n" % ("1" * (nExponent - 2),  "1" * nMantissa, nMantissa + 1)
        elif nExponent == 2: # the largest number < 1 is subnormal number When nExponent == 2
            ss += "| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{%s}$                     |  $1 - 2^{-%d}$        |  largest number < 1   |\n" % (                        "1" * nMantissa, nMantissa)
        else:
            # nExponent == 1
            ss += "| $\color{#FF0000}{0}\color{#007F00}{0}\color{#0000FF}{%s}$                      |  $1 - 2^{-%d}$        |  largest number < 1   |\n" % (                        "1" * nMantissa, nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{0%s}\color{#0000FF}{%s}$                        |  $1$                  |                       |\n" % ("1" * (nExponent - 1),  "0" * nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{0%s}\color{#0000FF}{%s1}$                       |  $1 + 2^{-%d}$        |  smallest number > 1  |\n" % ("1" * (nExponent - 1),  "0" * (nMantissa - 1), nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{1%s}\color{#0000FF}{%s}$                        |  $2$                  |                       |\n" % ("0" * (nExponent - 1),  "0" * nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{1%s}\color{#0000FF}{1%s}$                       |  $3$                  |                       |\n" % ("0" * (nExponent - 1),  "0" * (nMantissa - 1))
        if nExponent >= 3:
            # 4 can only be represented when nExponent >= 3
            ss += "| $\color{#FF0000}{0}\color{#007F00}{1%s1}\color{#0000FF}{%s}$                   |  $4$                  |                       |\n" % ("0" * (nExponent - 2),  "0" * nMantissa)
            if nMantissa >= 2:
                # 5 and 6 can only be represented when nExponent >= 3 and nMantissa >= 2
                ss += "| $\color{#FF0000}{0}\color{#007F00}{1%s1}\color{#0000FF}{01%s}$             |  $5$                  |                       |\n" % ("0" * (nExponent - 2),  "0" * (nMantissa - 2))
                ss += "| $\color{#FF0000}{0}\color{#007F00}{1%s1}\color{#0000FF}{10%s}$             |  $6$                  |                       |\n" % ("0" * (nExponent - 2),  "0" * (nMantissa - 2))
        if bIEEE754:
            ss += "| $\color{#FF0000}{0}\color{#007F00}{%s0}\color{#0000FF}{%s}$                    | $%s\\times10^{%s}$    |        Maximum        |\n" % ("1" * (nExponent - 1),  "1" * nMantissa,        *numberToString(        2 ** (2 ** (nExponent - 1) - 1) * (1 + (2 ** nMantissa - 1) / (2 ** nMantissa))))
            ss += "| $\color{#FF0000}{1}\color{#007F00}{%s0}\color{#0000FF}{%s}$                    | $%s\\times10^{%s}$    |     Maximum negtive   |\n" % ("1" * (nExponent - 1),  "1" * nMantissa,        *numberToString((-1) *  2 ** (2 ** (nExponent - 1) - 1) * (1 + (2 ** nMantissa - 1) / (2 ** nMantissa))))
        else:
            # exponent of all 1 is a normal value and at this occasion mantissa can not be all 1
            ss += "| $\color{#FF0000}{0}\color{#007F00}{\\bold{%s}}\color{#0000FF}{\\bold{%s0}}$    | $%s\\times10^{%s}$    |        Maximum        |\n" % ("1" * nExponent,        "1" * (nMantissa - 1),  *numberToString(        2 ** (2 ** (nExponent - 1)) * (1 + (2 ** nMantissa - 2) / (2 ** nMantissa))))
            ss += "| $\color{#FF0000}{1}\color{#007F00}{\\bold{%s}}\color{#0000FF}{\\bold{%s0}}$    | $%s\\times10^{%s}$    |     Maximum negtive   |\n" % ("1" * nExponent,        "1" * (nMantissa - 1),  *numberToString((-1) *  2 ** (2 ** (nExponent - 1)) * (1 + (2 ** nMantissa - 2) / (2 ** nMantissa))))
        ss += "| $\color{#FF0000}{1}\color{#007F00}{%s}\color{#0000FF}{%s}$                         | $-0$                  |                       |\n" % ("0" * nExponent,        "0" * nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{%s}$                         | $+\infty$             |   positive infinity   |\n" % ("1" * nExponent,        "0" * nMantissa)
        ss += "| $\color{#FF0000}{1}\color{#007F00}{%s}\color{#0000FF}{%s}$                         | $-\infty$             |   negative infinity   |\n" % ("1" * nExponent,        "0" * nMantissa)
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{%s1}$                        | $NaN$                 |         sNaN          |\n" % ("1" * nExponent,        "0" * (nMantissa - 1))
        if nMantissa >=2:
            # two kind of NaN can both be represented when nMantissa >= 2
            ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{1%s1}$                   | $NaN$                 |         qNaN          |\n" % ("1" * nExponent,        "0" * (nMantissa - 2))
        ss += "| $\color{#FF0000}{0}\color{#007F00}{%s}\color{#0000FF}{%s}$                         | $NaN$                 | other alternative NaN |\n" % ("1" * nExponent,        "1" * nMantissa)
        if nExponent >=3:
            # normal value less than 1/3 when nExponent >=3
            ss += "| $\color{#FF0000}{0}\color{#007F00}{0%s01}\color{#0000FF}{%s}$                 | $\\frac{1}{3}$        |                       |\n" % ("1" * (nExponent - 3),  "01" * (nMantissa // 2) + ("1" if nMantissa % 2 == 1 else ""))
        # yapf:enable
        f.write(ss)

    return

def convertBinaryToNumber(x, nSign, nExponent, nMantissa, nBias, bIEEE754):

    if len(x) != nSign + nExponent + nMantissa:
        print("Error input x")
        return None

    y = x
    if nSign > 0:
        sign = y[0]
        y = y[1:]
    else:
        sign = 0
    exponent = y[:nExponent]
    mantissa = y[nExponent:]

    # INF or NaN?

    # Normal or Subnormal?

    # TODO
    return

def convertNumberToBinary(x, sign, exponent, mantissa, bias, bIEEE754):
    # TODO
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", "-s", type=int, default=1, choices=[0, 1], help="count of sign bit, 0 or 1")
    parser.add_argument("--exponent", "-e", type=int, default=8, help="count of exponent bit, >= 2")
    parser.add_argument("--mantissa", "-m", type=int, default=23, help="count of mantissa bit, >= 1")
    parser.add_argument("--bias", "-b", type=int, default=0, help="bias")
    parser.add_argument("--bIEEE754", "-S", type=str, default="True", choices=["True", "False"], help="standard IEEE754")
    args = parser.parse_args()

    buildMD(args.sign, args.exponent, args.mantissa, args.bias, args.bIEEE754 == "True")
