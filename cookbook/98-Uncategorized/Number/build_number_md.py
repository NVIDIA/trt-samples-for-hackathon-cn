# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def binary_to_number(
    s_s: str = "",
    s_e: str = "",
    s_m: str = "",
    p: int = 0,
    q: int = 0,
    r: int = 0,
    b_keep_infinity: bool = True,
    b_keep_nan: bool = True,
):
    assert (len(s_s) == p)
    assert (len(s_e) == q)
    assert (len(s_m) == r)
    assert (set(s_s).issubset({'0', '1'}))
    assert (set(s_e).issubset({'0', '1'}))
    assert (set(s_m).issubset({'0', '1'}))

    b_IEEE754 = b_keep_infinity and b_keep_nan
    set0 = {"0"}
    set1 = {"1"}
    set01 = {"0", "1"}
    # Special values
    # if b_IEEE754:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Inifnity
    #   4. Exponent all 1, Mantissa not all 0 -> NaN
    # elif b_keep_nan:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Normal value, not special
    #   4. Exponent all 1, Mantissa not all 0 -> Normal value, not special
    #   5. Exponent all 1, Mantissa all 1     -> NaN
    # elif b_keep_infinity:
    #   No such case
    # else:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Normal value, not special
    #   4. Exponent all 1, Mantissa not all 0 -> Normal value, not special
    #   5. Exponent all 1, Mantissa all 1     -> Normal value, not special
    if set(s_e) == set0 and set(s_m) == set0:
        return 0, 0, 0
    elif b_IEEE754 and set(s_e) == set1 and set(s_m) == set0:
        return float("inf"), "", ""
    elif (b_IEEE754 and set(s_e) == set1 and set(s_m) == set01) or (b_keep_nan and set(s_e) == set1 and set(s_m) == set1):
        return float("nan"), "", ""
    elif not b_IEEE754 and b_keep_infinity:
        print("No such case")
        raise Exception

    i_s = int(s_s, 2) if len(s_s) > 0 else 0
    i_e = int(s_e, 2) if len(s_e) > 0 else 0
    i_m = int(s_m, 2) if len(s_m) > 0 else 0
    if "1" in s_e:  # Normal value
        e = i_e - (2 ** (q - 1) - 1)
        m = i_m * 2 ** (-r)
        value = (-1) ** i_s * 2 ** e * (m + 1)
    else:  # Subormal value
        e = 2 - 2 ** (q - 1)
        m = i_m * 2 ** (-r)
        value = (-1) ** i_s * 2 ** e * m
    a, b = f"{value:6.4e}".split("e")
    return value, a, b

def color_string(red_string, green_string, blue_string):
    return "$\color{#FF0000}{%s}\color{#007F00}{%s}\color{#0000FF}{%s}$" % (red_string, green_string, blue_string)

def case_q0(ss: str = "", p: int = 0, r: int = 0, b_keep_infinity: bool = True, b_keep_nan: bool = True):
    q = 0
    # Position Zero
    s_s, s_e, s_m = "0", "", "0" * r
    ss += "| %s | $+0$               | Positive Zero    |\n" % (color_string(s_s, s_e, s_m))

    # 1
    s_s, s_e, s_m = "0", "", "0" * (r - 1) + "1"
    ss += "| %s | $1$                |                  |\n" % (color_string(s_s, s_e, s_m))

    # Maximum
    s_s, s_e, s_m = "0", "", "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
    ss += "| %s | $%s\\times10^{%s}$ | Maximum |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Negative Maximum
    s_s, s_e, s_m = "1", "", "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
    ss += "| %s | $%s\\times10^{%s}$ | Maximum negative |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Negative Zero
    s_s, s_e, s_m = "1", "", "0" * r
    ss += "| %s | $-0$               | Positive Zero    |\n" % (color_string(s_s, s_e, s_m))

    # Print all non-negative values
    if p + 0 + r in [4, 6, 8]:
        ss += "\n+ All non-negative values\n\n"
        ss += "| Number(%s) | value |\n" % (color_string('Sign', 'Exponen', 'Mantissa'))
        ss += "| :-:        | :-:   |\n"

        for n in range(2 ** (p + q + r - 1)):
            binary = bin(n)[2:].zfill(p + q + r)
            s_s, s_e, s_m = binary[:p], binary[p:(p + q)], binary[(p + q):]
            value, _, _ = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
            #print(binary, s_s, s_e, s_m, value)
            ss += "| %s     | %.3e  |\n" % (color_string(s_s, s_e, s_m), value)
    return ss

def case_r0(ss: str = "", p: int = 0, q: int = 0, r: int = 0, b_keep_infinity: bool = True, b_keep_nan: bool = True):
    r = 0
    # Position Zero
    ss += "| None         | $0$               | Zero (not exist)       |\n"

    # Maximum
    s_s, s_e, s_m = "" * p, "1" * q, ""
    a, b = f"{2 ** (2 ** (q-1)):6.4e}".split("e")
    ss += "| %s     | $%s\\times10^{%s}$  | Maximum  |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Minimum
    s_s, s_e, s_m = "" * p, "0" * q, ""
    a, b = f"{-2 ** (2 ** (q-1)):6.4e}".split("e")
    ss += "| %s     | $%s\\times10^{%s}$ | Minimum |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Alternative NaN
    s_s, s_e, s_m = "" * p, "1" * q, ""
    ss += "| %s         | $NaN$              | NaN     |\n" % (color_string(s_s, s_e, s_m))

    # Normal value less than 1/3 can both be represented when q >= 3
    if q >= 3:
        s_s, s_e, s_m = "" * p, "0" + "1" * (q - 3) + "01", "01" * (r // 2) + ("1" if r % 2 == 1 else "")
        _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
        ss += "| %s     | $%s\\times10^{%s}$ | $\\frac{1}{3}$      |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Print all non-negative values for FP4, FP6, FP8
    if p + q + r in [4, 6, 8]:
        ss += "\n+ All non-negative values\n\n"
        ss += "| Number(%s) | value |\n" % (color_string('Sign', 'Exponen', 'Mantissa'))
        ss += "| :-:        | :-:   |\n"

        for n in range(2 ** (p + q + r)):
            binary = bin(n)[2:].zfill(p + q + r)
            s_s, s_e, s_m = binary[:p], binary[p:(p + q)], binary[(p + q):]
            value = 2 ** (int(s_e, 2) - (2 ** (q - 1) - 1))
            if n == 2 ** (p + q + r) - 1:
                value = float("nan")
            #print(binary, s_s, s_e, s_m, value)
            ss += "| %s     | %.3e  |\n" % (color_string(s_s, s_e, s_m), value)

    return ss

def case_general(ss: str = "", p: int = 0, q: int = 0, r: int = 0, b_keep_infinity: bool = True, b_keep_nan: bool = True):
    b_IEEE754 = b_keep_infinity and b_keep_nan

    # Position Zero
    s_s, s_e, s_m = "0", "0" * q, "0" * r
    ss += "| %s         | $+0$               | Positive Zero       |\n" % (color_string(s_s, s_e, s_m))

    # Smallest subnormal value
    s_s, s_e, s_m = "0", "0" * q, "0" * (r - 1) + "1"
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r)
    ss += "| %s         | $%s\\times10^{%s}$ | Minimum Subnormal   |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Largest subnormal value
    s_s, s_e, s_m = "0", "0" * q, "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r)
    ss += "| %s         | $%s\\times10^{%s}$ | Maximum Subnormal   |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Smallest normal value
    if q >= 1:  # Skip FP4E0M3
        s_s, s_e, s_m = "0", "0" * (q - 1) + "1", "0" * r
        _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r)
        ss += "| %s         | $%s\\times10^{%s}$ | Minimum Normal      |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Largest number < 1
    if q == 1:
        s_s, s_e, s_m = "0", "0", "1" * r
        ss += "| %s     | $1 - 2^{%d}$       | Largest Number < 1  |\n" % (color_string(s_s, s_e, s_m), -(r + 1))
    else:
        s_s, s_e, s_m = "0", "0" + "1" * (q - 2) + "0", "1" * r
        if q == 2 and r == 3:  # special case for FP6e2m3 since smallest normal value is 1
            ss += "| %s     | $1 - 2^{%d}$       | Largest Number < 1  |\n" % (color_string(s_s, s_e, s_m), -r)
        else:
            ss += "| %s     | $1 - 2^{%d}$       | Largest Number < 1  |\n" % (color_string(s_s, s_e, s_m), -(r + 1))

    # 1
    s_s, s_e, s_m = "0", "0" + "1" * (q - 1), "0" * r
    ss += "| %s         | $1$                |                     |\n" % (color_string(s_s, s_e, s_m))

    # Smallest number > 1
    s_s, s_e, s_m = "0", "0" + "1" * (q - 1), "0" * (r - 1) + "1"
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r)
    ss += "| %s         | $1 + 2^{%d}$       | Smallest Number > 1 |\n" % (color_string(s_s, s_e, s_m), -r)

    # 2
    s_s, s_e, s_m = "0", "1" + "0" * (q - 1), "0" * r
    ss += "| %s         | $2$                |                     |\n" % (color_string(s_s, s_e, s_m))

    # 3
    s_s, s_e, s_m = "0", "1" + "0" * (q - 1), "1" + "0" * (r - 1)
    ss += "| %s         | $3$                |                     |\n" % (color_string(s_s, s_e, s_m))

    # 4, 4 can only be represented when q >= 3
    if q >= 3:
        s_s, s_e, s_m = "0", "1" + "0" * (q - 2) + "1", "0" * r
        ss += "| %s     | $4$                |                     |\n" % (color_string(s_s, s_e, s_m))

    # 5 and 6 can only be represented when q >= 3 and r >= 2
    # 5
    if q >= 3 and r >= 2:
        s_s, s_e, s_m = "0", "1" + "0" * (q - 2) + "1", "01" + "0" * (r - 2)
        ss += "| %s     | $5$                |                     |\n" % (color_string(s_s, s_e, s_m))
    # 6
    if q >= 3 and r >= 2:
        s_s, s_e, s_m = "0", "1" + "0" * (q - 2) + "1", "10" + "0" * (r - 2)
        ss += "| %s     | $6$                |                     |\n" % (color_string(s_s, s_e, s_m))

    # Maximum
    if b_IEEE754:
        s_s, s_e, s_m = "0", "1" * (q - 1) + "0", "1" * r
    else:
        if b_keep_nan:  # exponent all 1 is normal value, and mantissa can not be all 1
            s_s, s_e, s_m = "0", "1" * q, "1" * (r - 1) + "0"
        else:  # exponent all 1 is normal value, and mantissa can be all 1
            s_s, s_e, s_m = "0", "1" * q, "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
    ss += "| %s     | $%s\\times10^{%s}$ | Maximum |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Negative Maximum
    if b_IEEE754:
        s_s, s_e, s_m = "1", "1" * (q - 1) + "0", "1" * r
    else:
        if b_keep_nan:  # exponent all 1 is normal value, and mantissa can not be all 1
            s_s, s_e, s_m = "1", "1" * q, "1" * (r - 1) + "0"
        else:  # exponent all 1 is normal value, and mantissa can be all 1
            s_s, s_e, s_m = "1", "1" * q, "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
    ss += "| %s     | $%s\\times10^{%s}$ | Maximum negative    |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Negative Zero
    s_s, s_e, s_m = "1", "0" * q, "0" * r
    ss += "| %s         | $-0$               | Negative Zero       |\n" % (color_string(s_s, s_e, s_m))

    if b_keep_infinity:
        # Positive Infinity
        s_s, s_e, s_m = "0", "1" * q, "0" * r
        ss += "| %s     | $+\infty$          | Positive Infinity   |\n" % (color_string(s_s, s_e, s_m))

        # Negative Inifity
        s_s, s_e, s_m = "1", "1" * q, "0" * r
        ss += "| %s     | $-\infty$          | Negative Infinity   |\n" % (color_string(s_s, s_e, s_m))

    if b_IEEE754:
        # Signalling Not a Number
        s_s, s_e, s_m = "0", "1" * q, "0" * (r - 1) + "1"
        ss += "| %s     | $NaN$              | Signalling NaN      |\n" % (color_string(s_s, s_e, s_m))

        # another kind of NaN can both be represented when r >= 2
        if r >= 2:
            # Quiet Not a Number
            s_s, s_e, s_m = "0", "1" * q, "1" + "0" * (r - 2) + "1"
            ss += "| %s | $NaN$              | Quiet NaN           |\n" % (color_string(s_s, s_e, s_m))

    if b_keep_nan:
        # Alternative NaN
        s_s, s_e, s_m = "0", "1" * q, "1" * r
        ss += "| %s         | $NaN$              | NaN                 |\n" % (color_string(s_s, s_e, s_m))

    # Normal value less than 1/3 can both be represented when q >= 3
    if q >= 3:
        s_s, s_e, s_m = "0", "0" + "1" * (q - 3) + "01", "01" * (r // 2) + ("1" if r % 2 == 1 else "")
        _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
        ss += "| %s     | $%s\\times10^{%s}$ | $\\frac{1}{3}$      |\n" % (color_string(s_s, s_e, s_m), a, b)

    # Print all values for FP4, FP6, FP8
    if p + q + r in [4, 6, 8]:
        ss += "\n+ All non-negative values\n\n"
        ss += "| Number(%s) | value |\n" % (color_string('Sign', 'Exponen', 'Mantissa'))
        ss += "| :-:        | :-:   |\n"

        for n in range(2 ** (p + q + r - 1)):
            binary = bin(n)[2:].zfill(p + q + r)
            s_s, s_e, s_m = binary[:p], binary[p:(p + q)], binary[(p + q):]
            value, _, _ = binary_to_number(s_s, s_e, s_m, p, q, r, b_keep_infinity, b_keep_nan)
            #print(binary, s_s, s_e, s_m, value)
            ss += "| %s     | %.3e  |\n" % (color_string(s_s, s_e, s_m), value)

    return ss

def build_md(p: int = 0, q: int = 0, r: int = 0):
    # p: number bits of sign
    # q: number bits of exponent
    # r: number bits of mantissa

    type_name = f"S{p}E{q}M{r}"
    b_keep_infinity = True
    b_keep_nan = True
    if type_name == "S1E4M3":
        type_name = "FP8e4m3"
        b_keep_infinity = False
    elif type_name == "S1E5M2":
        type_name = "FP8e5m2"
    elif type_name == "S1E5M10":
        type_name = "FP16"
    elif type_name == "S1E8M7":
        type_name = "BF16"
    elif type_name == "S1E8M10":
        type_name = "TF32"
    elif type_name == "S1E8M23":
        type_name = "FP32"
    if type_name == "S1E11M52":
        type_name = "FP64"
    elif type_name == "S1E3M2":
        type_name = "FP6e3m2"
        b_keep_infinity = False
        b_keep_nan = False
    elif type_name == "S1E2M3":
        type_name = "FP6e2m3"
        b_keep_infinity = False
        b_keep_nan = False
    elif type_name == "S1E2M1":
        type_name = "FP4e2m1(MXFP4)"
        b_keep_infinity = False
        b_keep_nan = False
    elif type_name == "S1E0M3":  # TODO: need check
        type_name = "FP4e0m3(NVFP4)"
        b_keep_infinity = False
        b_keep_nan = False
    elif type_name == "S0E8M0":  # TODO: need check
        type_name = "FP8e8m0"
        b_keep_infinity = False
        b_keep_nan = False

    print(f"Build {type_name}")

    ss = ""
    ss += f"# {type_name} - {'' if b_keep_infinity and b_keep_nan else 'not '}IEEE754\n"
    ss += "\n"
    ss += f"+ Sign:     $s$ ($p={p}$ bit)\n"
    ss += f"+ Exponent: $e$ ($q={q}$ bits)\n"
    ss += f"+ Mantissa: $m$ ($r={r}$ bits)\n"
    ss += "\n"

    ss += "+ Normal value ($%s1_2 \le e_2 \le %s0_2$) (subscript 2 represents the base)\n" % ('0' * (q - 1), '1' * (q - 1))
    ss += "\n$$\n"
    ss += "\\begin{equation}\n"
    ss += "\\begin{aligned}\n"
    ss += "E &= e - \left( 2^{q-1} - 1 \\right) = e - %d \\\\\n" % (2 ** (q - 1) - 1)
    ss += "M &= m \cdot 2^{-r} = m \cdot 2^{%d} \\\\\n" % (-r)
    ss += "value &= \left( -1 \\right) ^ {s} 2 ^ {E} \left( 1 + M \\right) = \left( -1 \\right) ^ {s} 2 ^ {%d} 2 ^ {e} \left( m  + 2^{%d} \\right)\n" % (-(2 ** (q - 1) - 1) - r, r)
    ss += "\end{aligned}\n"
    ss += "\end{equation}\n"
    ss += "$$\n"
    ss += "\n"

    ss += "+ Subnormal value ($e_2 = %s_2$)\n" % ('0' * q)
    ss += "\n$$\n"
    ss += "\\begin{equation}\n"
    ss += "\\begin{aligned}\n"
    ss += "E &= 2 - 2^{q-1} = %d \\\\\n" % (2 - 2 ** (q - 1))
    ss += "M &= m \cdot 2^{-r} = m \cdot 2^{%d} \\\\\n" % (-r)
    ss += "value &= \left( -1 \\right) ^ {s} 2 ^ {E} M = \left( -1 \\right) ^ {s} 2^{%d} m\n" % (2 - 2 ** (q - 1))
    ss += "\end{aligned}\n"
    ss += "\end{equation}\n"
    ss += "$$\n"
    ss += "\n"

    ss += "+ Special value\n\n"
    ss += "| Exponent  | Mantissa  | meaning         |\n"
    ss += "| :-:       | :-:       | :-:             |\n"
    ss += "| all 0     | all 0     | Signed Zero     |\n"
    ss += "| all 0     | not all 0 | Subnormal Value |\n"
    if b_keep_infinity:
        ss += "| all 1 | all 0     | Signed Infinity |\n"
    else:
        ss += "| all 1 | all 0     | Normal value    |\n"
    if b_keep_nan:
        ss += "| all 1 | not all 0 | qNaN, sNaN      |\n"
    else:
        ss += "| all 1 | not all 0 | Normal value    |\n"
    if p == 0 and r == 0:  # Special case for FP8e8m0
        ss += "| all 1 | all 1     | NaN             |\n"
    ss += "\n"

    ss += "+ Examples\n\n"
    ss += "| Number(%s) | value          | comment          |\n" % (color_string('Sign', 'Exponent', 'Mantissa'))
    ss += "| :-:        | :-:            | :-:              |\n"

    if q == 0:
        ss = case_q0(ss, p, r, b_keep_infinity, b_keep_nan)
    elif r == 0:
        ss = case_r0(ss, p, q, b_keep_infinity, b_keep_nan)
    else:
        ss = case_general(ss, p, q, r, b_keep_infinity, b_keep_nan)

    with open(f"output/{type_name}.md", "w") as f:
        f.write(ss)

    return

if __name__ == "__main__":
    build_md(1, 4, 3)  # FP8e4m3-not-IEEE754
    build_md(1, 5, 2)  # FP8e5m2
    build_md(1, 5, 10)  # FP16
    build_md(1, 8, 7)  # BF16
    build_md(1, 8, 10)  # TF32
    build_md(1, 8, 23)  # FP32
    build_md(1, 11, 52)  # FP64
    build_md(1, 2, 3)  # FP6e2m3
    build_md(1, 3, 2)  # FP6e3m2
    build_md(1, 2, 1)  # FP4e2m1
    build_md(1, 0, 3)  # FP4e0m3
    build_md(0, 8, 0)  # Scale Type

    print("Finish")
