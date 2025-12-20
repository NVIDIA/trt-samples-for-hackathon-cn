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

import math
from typing import Dict
from dataclasses import dataclass
from enum import Enum

class FloatFormat(Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    TF32 = "TF32"
    FP64 = "FP64"
    FP8E4M3 = "FP8E4M3"
    FP8E5M2 = "FP8E5M2"
    FP8E8M0 = "FP8E8M0"
    FP6E3M2 = "FP6E3M2"
    FP6E2M3 = "FP6E2M3"
    FP4E2M1 = "FP4E2M1"
    FP4E0M3 = "FP4E0M3"

@dataclass
class FloatFormatSpec:
    name: str
    total_bits: int
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    has_implicit_bit: bool = True  # Whether subnormal mode is supported
    has_inf: bool = True
    has_nan: bool = True
    is_ieee754: bool = True  # In case that a data type has inf and nan but not follows IEEE 754 standard

class FloatConverter:
    """
    FloatConverter provides methods to convert between custom floating-point binary string representations and Python float values, supporting various floating-point formats.

    Attributes:
        format_specs (dict): A mapping from FloatFormat to their corresponding format specifications.
        fp4e0m3_lookup (dict): Lookup table for FP4E0M3 format to map mantissa bits to float values.

    Methods:
        binary_string_to_float(binary_string: str, float_format: FloatFormat) -> float:
            Converts a binary string representation of a floating-point number to a Python float, according to the specified format.

        float_to_binary_string(value: float, float_format: FloatFormat) -> str:
            Converts a Python float to its binary string representation in the specified floating-point format.

        convert_between_formats(binary_string: str, from_format: FloatFormat, to_format: FloatFormat) -> str:
            Converts a binary string from one floating-point format to another by first decoding to float and then encoding to the target format.

        get_format_info(float_format: FloatFormat) -> Dict:
            Returns a dictionary containing the specification details for the given floating-point format.

    Private Methods:
        _float_to_fp8e8m0(value: float) -> str:
            Converts a float to the FP8E8M0 binary string representation.

        _float_to_fp4e0m3(value: float) -> str:
            Converts a float to the FP4E0M3 binary string representation.

        _get_max_normal_binary(float_format: FloatFormat) -> str:
            Returns the binary string representation of the maximum normal number for the specified format.

    + 4 styles of representing the float number:
      + Decimal value:          value=314.159265
      + Sem2 expression:        s=0, e=10000111, m=00111010001010001100011
      + Sem10 expression:       s=0, e=135, m=1905763
      + Scientific expression:  x=3.14159265, y=2
    """

    def __init__(self):
        self.format_specs = {
            FloatFormat.FP32: FloatFormatSpec("FP32", 32, 1, 8, 23),
            FloatFormat.FP16: FloatFormatSpec("FP16", 16, 1, 5, 10),
            FloatFormat.BF16: FloatFormatSpec("BF16", 16, 1, 8, 7),
            FloatFormat.TF32: FloatFormatSpec("TF32", 19, 1, 8, 10),
            FloatFormat.FP64: FloatFormatSpec("FP64", 64, 1, 11, 52),
            FloatFormat.FP8E4M3: FloatFormatSpec("FP8E4M3", 8, 1, 4, 3, has_inf=False, is_ieee754=False),
            FloatFormat.FP8E5M2: FloatFormatSpec("FP8E5M2", 8, 1, 5, 2),
            FloatFormat.FP8E8M0: FloatFormatSpec("FP8E8M0", 8, 0, 8, 0, has_inf=False, has_nan=False, has_implicit_bit=False, is_ieee754=False),
            FloatFormat.FP6E3M2: FloatFormatSpec("FP6E3M2", 6, 1, 3, 2, has_inf=False, has_nan=False, is_ieee754=False),
            FloatFormat.FP6E2M3: FloatFormatSpec("FP6E2M3", 6, 1, 2, 3, has_inf=False, has_nan=False, is_ieee754=False),
            FloatFormat.FP4E2M1: FloatFormatSpec("FP4E2M1", 4, 1, 2, 1, has_inf=False, has_nan=False, is_ieee754=False),
            FloatFormat.FP4E0M3: FloatFormatSpec("FP4E0M3", 4, 1, 0, 3, has_inf=False, has_nan=False, has_implicit_bit=False, is_ieee754=False),
        }
        self.fp4e0m3_lookup = {0: 0.0, 1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875}

    def binary_string_to_float(self, binary_string: str, float_format: FloatFormat) -> float:
        spec = self.format_specs[float_format]

        if len(binary_string) != spec.total_bits:
            raise ValueError(f"{float_format.value} requires {spec.total_bits} bits, got {len(binary_string)}")
        if len(set(binary_string) - {'0', '1'}) > 0:
            raise ValueError("Binary string must contain only 0 and 1")

        # Special formats
        if float_format == FloatFormat.FP8E8M0:
            return 2.0 ** (int(binary_string, 2) - (2 ** (spec.exponent_bits - 1) - 1))
        elif float_format == FloatFormat.FP4E0M3:
            sign_bit = int(binary_string[0], 2)
            mantissa = int(binary_string[1:], 2)
            value = self.fp4e0m3_lookup.get(mantissa, 0.0)
            return value if sign_bit == 0 else -value

        # Standard process
        sign_bit = int(binary_string[0], 2)
        exponent_bits = binary_string[1:1 + spec.exponent_bits]
        mantissa_bits = binary_string[1 + spec.exponent_bits:]

        sign = 1 if sign_bit == 0 else -1
        exponent = int(exponent_bits, 2)
        mantissa = int(mantissa_bits, 2)

        # inf and nan
        if exponent == (1 << spec.exponent_bits) - 1:
            if spec.is_ieee754 or (spec.has_inf and spec.has_nan):
                if mantissa == 0:
                    return float('inf') if sign_bit == 0 else float('-inf')
                else:
                    return float('nan')
            elif float_format == FloatFormat.FP8E4M3 or (spec.has_nan and not spec.has_inf):
                if mantissa == (1 << spec.mantissa_bits) - 1:
                    return float('nan')

        if exponent == 0:  # 0 or subnormal number
            if mantissa == 0:
                return 0.0 * sign
            else:
                return sign * (2 ** (2 - 2 ** (spec.exponent_bits - 1))) * mantissa / (1 << spec.mantissa_bits)
        else:  # Normal number
            return sign * 2 ** (exponent - (2 ** (spec.exponent_bits - 1) - 1)) * (mantissa / (1 << spec.mantissa_bits) + int(spec.has_implicit_bit))

    def _float_to_fp8e8m0(self, value: float) -> str:
        if math.isnan(value) or value >= 2 ** 128:
            return '11111111'
        elif value <= 0:
            return '00000000'
        true_exponent = int(round(math.log2(value)))
        true_exponent = max(-127, min(127, true_exponent))
        biased_exponent = true_exponent + (2 ** (8 - 1) - 1)
        return format(biased_exponent, '08b')  # Convert integer to binary string

    def _float_to_fp4e0m3(self, value: float) -> str:
        if value == 0.0:
            return '0000'
        sign_bit = '1' if value < 0 else '0'
        abs_value = abs(value)
        # Find the closest value in the lookup table
        closest_distance = float('inf')
        closest_mantissa = 0
        for mantissa, table_value in self.fp4e0m3_lookup.items():
            distance = abs(abs_value - abs(table_value))
            if distance < closest_distance:
                closest_distance = distance
                closest_mantissa = mantissa
        return sign_bit + format(closest_mantissa, '03b')

    def float_to_binary_string(self, value: float, float_format: FloatFormat) -> str:
        spec = self.format_specs[float_format]

        # Special formats
        if float_format == FloatFormat.FP8E8M0:
            return self._float_to_fp8e8m0(value)
        elif float_format == FloatFormat.FP4E0M3:
            return self._float_to_fp4e0m3(value)

        if math.isnan(value):
            if not spec.has_nan:
                return self._get_max_normal_binary(float_format)
            return '0' + '1' * (spec.exponent_bits + spec.mantissa_bits)

        if math.isinf(value):
            sign_bit = ('1' if value < 0 else '0')
            if not spec.has_inf:
                return sign_bit + self._get_max_normal_binary(float_format)[1:]
            return sign_bit + '1' * spec.exponent_bits + '0' * spec.mantissa_bits

        if value == 0.0:
            return ('1' if math.copysign(1, value) < 0 else '0') + '0' * (spec.total_bits - 1)

        sign_bit = '1' if value < 0 else '0'
        abs_value = abs(value)

        if abs_value >= 1:
            exponent = int(math.floor(math.log2(abs_value)))
        else:
            exponent = int(math.ceil(math.log2(abs_value)))

        maximum_binary = self._get_max_normal_binary(float_format)
        maximum = self.binary_string_to_float(maximum_binary, float_format)
        zero = sign_bit + '0' * (spec.total_bits - 1)  # 0
        if abs_value > maximum:  # Overflow to max normal number
            return maximum_binary
        elif exponent < 2 - 2 ** (spec.exponent_bits - 1):  # 0 or subnormal number
            if not spec.has_implicit_bit:
                return zero
            significand = abs_value / (2 ** (2 - 2 ** (spec.exponent_bits - 1)))
            if significand >= 1:
                return zero
            mantissa = int(round(significand * (1 << spec.mantissa_bits)))
            if mantissa == 0:
                return zero
            return sign_bit + '0' * spec.exponent_bits + format(mantissa, f'0{spec.mantissa_bits}b')

        # Normal number
        exponent_bits = format(exponent + (2 ** (spec.exponent_bits - 1) - 1), f'0{spec.exponent_bits}b')

        significand = abs_value / (2 ** exponent) - int(spec.has_implicit_bit)
        mantissa = int(round(significand * (1 << spec.mantissa_bits)))
        mantissa = max(0, min(mantissa, (1 << spec.mantissa_bits) - 1))
        mantissa_bits = format(mantissa, f'0{spec.mantissa_bits}b')

        return sign_bit + exponent_bits + mantissa_bits

    def _get_max_normal_binary(self, float_format: FloatFormat) -> str:
        if float_format == FloatFormat.FP8E8M0:
            return '01111111'
        elif float_format == FloatFormat.FP4E0M3:
            return '0111'
        elif float_format == FloatFormat.FP8E4M3:
            return '01111110'

        spec = self.format_specs[float_format]
        if spec.is_ieee754 or (spec.has_inf or spec.has_nan):
            max_exponent = (1 << spec.exponent_bits) - 2
        else:
            max_exponent = (1 << spec.exponent_bits) - 1
        return '0' + format(max_exponent, f'0{spec.exponent_bits}b') + '1' * spec.mantissa_bits

    def convert_between_formats(self, binary_string: str, from_format: FloatFormat, to_format: FloatFormat) -> str:
        float_value = self.binary_string_to_float(binary_string, from_format)
        return self.float_to_binary_string(float_value, to_format)

    def get_format_info(self, float_format: FloatFormat) -> Dict:
        spec = self.format_specs[float_format]
        return {name: getattr(spec, name) for name in spec.__dataclass_fields__.keys()}

# Unit tests
def test_converter():
    print("=== Unit tests for FloatConverter ===\n")
    converter = FloatConverter()

    for index, float_format in enumerate(FloatFormat):
        info = converter.get_format_info(float_format)
        print(f"{index:2d} -> {float_format.value}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

    test_cases = [0, 0.5, 1.0, 3.14, 10.75, 4.0e38, -0.0, -0.5, -1.0, -10.75, float('inf'), float('-inf'), float('nan')]
    # 10.75_10 == 1010.11_2

    for float_format in FloatFormat:
        print(f"=== Test {float_format.value} ===")
        for value in test_cases:
            print(f"{value:16e}", end=' ->')
            binary_str = converter.float_to_binary_string(value, float_format)
            print(f"{binary_str:32s}", end=' ->')
            converted_back = converter.binary_string_to_float(binary_str, float_format)
            print(f"{converted_back:16e}")

# Deprecated
def binary_to_number(
    s_s: str = "",
    s_e: str = "",
    s_m: str = "",
    p: int = 0,
    q: int = 0,
    r: int = 0,
    has_inf: bool = True,
    has_nan: bool = True,
):
    assert (len(s_s) == p)
    assert (len(s_e) == q)
    assert (len(s_m) == r)
    assert (set(s_s).issubset({'0', '1'}))
    assert (set(s_e).issubset({'0', '1'}))
    assert (set(s_m).issubset({'0', '1'}))

    is_ieee754 = has_inf and has_nan
    set0 = {"0"}
    set1 = {"1"}
    set01 = {"0", "1"}
    # Special values
    # if is_ieee754:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Inifnity
    #   4. Exponent all 1, Mantissa not all 0 -> NaN
    # elif has_nan:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Normal value, not special
    #   4. Exponent all 1, Mantissa not all 0 -> Normal value, not special
    #   5. Exponent all 1, Mantissa all 1     -> NaN
    # elif has_inf:
    #   No such case
    # else:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Normal value, not special
    #   4. Exponent all 1, Mantissa not all 0 -> Normal value, not special
    #   5. Exponent all 1, Mantissa all 1     -> Normal value, not special
    if set(s_e) == set0 and set(s_m) == set0:
        return 0, 0, 0
    elif is_ieee754 and set(s_e) == set1 and set(s_m) == set0:
        return float("inf"), "", ""
    elif (is_ieee754 and set(s_e) == set1 and set(s_m) == set01) or (has_nan and set(s_e) == set1 and set(s_m) == set1):
        return float("nan"), "", ""
    elif not is_ieee754 and has_inf:
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

# Generate Latex in Markdown:
# For variables for evaluation but not need "{}" in Latex, use "{vv}"
# For variables not for evaluation but need "{}" in Latex, use "{{vv}}"
# For variables both for evaluation and need "{}" in Latex, use "{{{vv}}}"

normal_value_template = r"""+ Normal value (${normal_exponent_min}_2 \le e_2 \le {normal_exponent_max}_2$) (2 in subscript represents the base)

$$
\begin{{equation}}
\begin{{aligned}}
E &= e - \left( 2 ^ {{q-1}} - 1 \right) = e - {normal_bias} \\
M &= m \cdot 2 ^ {{-r}} = m \cdot 2 ^ {{-{r}}} \\
\text{{value}} &= \left( -1 \right) ^ {{s}} 2 ^ {{E}} \left( 1 + M \right) = \left( -1 \right) ^ {{s}} 2 ^ {{{neg_r_minus_normal_bias}}} 2 ^ {{e}} \left( m + 2 ^ {{{r}}} \right)
\end{{aligned}}
\end{{equation}}
$$

"""

subnormal_value_template = r"""+ Subnormal value ($e_2 = {subnormal_exponent_zeros}_2$)

$$
\begin{{equation}}
\begin{{aligned}}
E &= 2 - 2 ^ {{q-1}} = -{subnormal_bias} \\
M &= m \cdot 2 ^ {{-r}} = m \cdot 2 ^ {{-{r}}} \\
\text{{value}} &= \left( -1 \right) ^ {{s}} 2 ^ {{E}} M = \left( -1 \right) ^ {{s}} 2 ^ {{{neg_r_minus_subnormal_bias}}} m
\end{{aligned}}
\end{{equation}}
$$

"""

q0_value_template = r"""+ Value

$$
\begin{{equation}}
\begin{{aligned}}
E &= 0 \\
M &= m \cdot 2 ^ {{-r}} = m \cdot 2 ^ {{-{r}}} \\
\text{{value}} &= \left( -1 \right) ^ {{s}} 2 ^ {{E}} M = \left( -1 \right) ^ {{s}} m \cdot 2 ^ {{-{r}}}
\end{{aligned}}
\end{{equation}}
$$

"""

fp8e8m0_value_template = r"""+ Value

$$
\begin{{equation}}
\begin{{aligned}}
E &= e - \left( 2 ^ {{q-1}} - 1 \right) = e - {normal_bias} \\
M &= 1 \\
\text{{value}} &= 2 ^ {{E}} M = 2 ^ {{e - {normal_bias}}}
\end{{aligned}}
\end{{equation}}
$$

"""

def build_md(float_format: FloatFormat):
    print(f"Build {float_format.value}.md")

    converter = FloatConverter()

    spec = converter.format_specs[float_format]
    p, q, r = spec.sign_bits, spec.exponent_bits, spec.mantissa_bits
    has_inf = spec.has_inf
    has_nan = spec.has_nan
    is_ieee754 = spec.is_ieee754

    comment = ""
    if float_format == FloatFormat.FP4E2M1:
        comment = " (MXFP4 or NVFP4)"

    ss = ""
    ss += f"# {float_format.value}{comment} - {'' if is_ieee754 else 'not '}IEEE754\n"
    ss += "\n"
    ss += f"+ Sign:     $s$ ($p = {p}$ bit)\n"
    ss += f"+ Exponent: $e$ ($q = {q}$ bit)\n"
    ss += f"+ Mantissa: $m$ ($r = {r}$ bit)\n"
    ss += "\n"

    if float_format == FloatFormat.FP8E8M0:
        ss += fp8e8m0_value_template.format(normal_bias=2 ** (8 - 1) - 1)
    elif float_format == FloatFormat.FP4E0M3:
        ss += q0_value_template.format(r=r)
    else:
        ss += normal_value_template.format(
            normal_exponent_min=format(1, f'0{q}b'),
            normal_exponent_max=format(2 ** q - 2, f'0{q}b'),
            normal_bias=2 ** (q - 1) - 1,
            r=r,
            neg_r_minus_normal_bias=-r - 2 ** (q - 1) + 1,
        )

        ss += subnormal_value_template.format(
            subnormal_exponent_zeros=format(0, f'0{q}b'),
            subnormal_bias=2 ** (q - 1) - 2,
            r=r,
            neg_r_minus_subnormal_bias=-r - 2 ** (q - 1) + 2,
        )

    if q > 0 and r > 0:  # Otherwise, printing all values is enough

        ss += "+ Special value\n"
        if not has_inf:
            ss += "  - **No bits pattern for infinity.**\n"
        if not has_nan:
            ss += "  - **No bits pattern for NaN.**\n"
        ss += "\n"
        ss += "| Exponent | Mantissa | meaning |\n"
        ss += "|:-:|:-:|:-:|\n"
        ss += "| all 0 | all 0 | Signed Zero |\n"
        ss += "| all 0 | not all 0 | Subnormal Value |\n"
        if has_inf:
            ss += "| all 1 | all 0 | Signed Infinity |\n"
        else:
            ss += "| all 1 | all 0 | Normal value |\n"

        if float_format == FloatFormat.FP8E4M3:
            ss += "| all 1 | not all 1 | Normal value |\n"
            ss += "| all 1 | all 1 | NaN |\n"
        elif has_nan:
            ss += "| all 1 | not all 0 | NaN |\n"
        else:
            ss += "| all 1 | not all 0 | Normal value |\n"
        ss += "\n"

        def integer_to_binary(value: int = 0, width: int = 1):
            return format(value, f'0{width}b')

        def sem10_to_bits_tuple(x: list):
            return integer_to_binary(x[0], spec.sign_bits), integer_to_binary(x[1], spec.exponent_bits), integer_to_binary(x[2], spec.mantissa_bits)

        def sem10_to_bits(x: list):
            return "".join(sem10_to_bits_tuple(x))

        def rgb(red_string, green_string, blue_string):
            return r"$\color{#D62728}{%s}\color{#2CA02C}{%s}\color{#1F77B4}{%s}$" % (red_string, green_string, blue_string)

        def sem10_to_rgb_bits(x: list):
            return rgb(*sem10_to_bits_tuple(x))

        def get_format_dict(sem10: list, comment: str, special_value_string: str = ""):
            if special_value_string == "":
                m, e = f"{converter.binary_string_to_float(sem10_to_bits(sem10), float_format):8.6e}".split("e")
            return {
                "rgb_string": sem10_to_rgb_bits(sem10),
                "value_string": f"${m}\\times10^{{{e}}}$" if special_value_string == "" else special_value_string,
                "comment": comment,
            }

    if q > 0 and r > 0:  # Otherwise, printing all values is enough

        ss += "+ Examples\n\n"
        ss += f"| Number({rgb('Sign', 'Exponent', 'Mantissa')}) | value | comment |\n"
        ss += "|:-:|:-:|:-:|\n"

        ss_template = r"| {rgb_string} | {value_string} | {comment} |" + "\n"

        # Position Zero
        ss += ss_template.format(**get_format_dict([0, 0, 0], "Positive Zero", "$+0$"))

        # Smallest subnormal value
        ss += ss_template.format(**get_format_dict([0, 0, 1], "Minimum Subnormal"))

        # Largest subnormal value
        ss += ss_template.format(**get_format_dict([0, 0, 2 ** r - 1], "Maximum Subnormal"))

        # Smallest normal value
        if q >= 1:
            ss += ss_template.format(**get_format_dict([0, 1, 0], "Minimum Normal"))
        else:
            ss += ss_template.format(**get_format_dict([0, 0, 1], "Minimum Normal"))  ############

        # Largest number < 1
        if q == 1:  # For FP4E1M2
            s_s, s_e, s_m = "0", "0", "1" * r
            ss += "| %s     | $1 - 2^{%d}$       | Largest number < 1  |\n" % (rgb(s_s, s_e, s_m), -(r + 1))
            raise Exception("Unchecked")
        else:
            s_s, s_e, s_m = "0", "0" + "1" * (q - 2) + "0", "1" * r
            if q == 2 and r == 3:  # FP6E2M3 since smallest normal value is 1
                ss += "| %s     | $1 - 2^{%d}$       | Largest number < 1  |\n" % (rgb(s_s, s_e, s_m), -r)
            else:
                ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 2, 2 ** r - 1], "Largest number < 1", f"$1 - 2 ^ {{-{r+1}}}$"))

        # 1
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 1, 0], "", "$1$"))

        # Smallest number > 1
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 1, 1], "Smallest number > 1", f"$1 + 2 ^ {{-{r}}}$"))

        # 2
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1), 0], "", "$2$"))

        # 3
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1), 2 ** (r - 1)], "", "$3$"))

        # 4
        if q >= 2:
            ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) + 1, 0], "", "$4$"))

        # 5
        if q >= 2 and r >= 2:
            ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) + 1, 2 ** (r - 2)], "", "$5$"))
        # 6
        if q >= 2 and r >= 2:
            ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) + 1, 2 ** (r - 1)], "", "$6$"))

        # Maximum
        if is_ieee754:
            s_s, s_e, s_m = "0", "1" * (q - 1) + "0", "1" * r
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 2, 2 ** r - 1], "Positive maximum"))
        elif float_format == FloatFormat.FP8E4M3:  # has no inf but has nan
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 2 ** r - 2], "Positive maximum"))
        elif float_format in [FloatFormat.FP6E2M3, FloatFormat.FP6E3M2, FloatFormat.FP4E2M1]:  # has no inf or nan
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 2 ** r - 1], "Positive maximum"))

        # Negative Maximum
        if is_ieee754:
            s_s, s_e, s_m = "1", "1" * (q - 1) + "0", "1" * r
            ss += ss_template.format(**get_format_dict([1, 2 ** q - 2, 2 ** r - 1], "Negative maximum"))
        elif float_format == FloatFormat.FP8E4M3:  # has no inf but has nan
            ss += ss_template.format(**get_format_dict([1, 2 ** q - 1, 2 ** r - 2], "Negative maximum"))
        elif float_format in [FloatFormat.FP6E2M3, FloatFormat.FP6E3M2, FloatFormat.FP4E2M1]:  # has no inf or nan
            ss += ss_template.format(**get_format_dict([1, 2 ** q - 1, 2 ** r - 1], "Negative maximum"))

        # Negative Zero
        ss += ss_template.format(**get_format_dict([1, 0, 0], "Negative Zero", "$-0$"))

        if has_inf:
            # Positive Infinity
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 0], "Positive Infinity", r"$+\infty$"))

            # Negative Infinity
            ss += ss_template.format(**get_format_dict([1, 2 ** q - 1, 0], "Negative Infinity", r"$-\infty$"))

        if is_ieee754:
            # Signalling Not a Number
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 1], "Signalling NaN", "sNaN"))

            # another kind of NaN can both be represented when r >= 2
            if r >= 2:
                # Quiet Not a Number
                ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 2 ** (r - 1) + 1], "Quiet NaN", "qNaN"))

        if has_nan:
            # Alternative NaN
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 2 ** r - 1], "NaN", "NaN"))

        # Normal value less than 1/3 can both be represented when q >= 3
        if q >= 3:
            rr = int((2 ** r - 1) / 3) if r % 2 == 0 else int((2 ** r + 1) / 3)
            ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 3, rr], "$\\frac{1}{3}$"))

    # Print all values for FP4, FP6, FP8
    if spec.total_bits in [4, 6, 8]:
        ss += "\n"

        if p == 0:
            ss += f"**+ No sign bit for data type {float_format.name}, ignore the S bit below.**\n"

        line = r"|$\color{{#D62728}}{{S}}$|$\color{{#2CA02C}}{{E}}$|$\color{{#1F77B4}}{{M={m}}}$|".format(m=format(0, f'0{r}b'))
        for i in range(1, 2 ** r):
            line += r"$\color{{#1F77B4}}{m}$|".format(m=format(i, f'0{r}b'))
        ss += line + "\n"

        line = "|:-:|:-:|" + "-:|" * 2 ** r + "\n"
        ss += line

        for s in range(2 ** p):
            for e in range(2 ** q):
                line = r"|$\color{{#D62728}}{{{s}}}$|$\color{{#2CA02C}}{{{e}}}$|".format(s=s, e=format(e, f'0{q}b'))
                for m in range(2 ** r):
                    line += f"{converter.binary_string_to_float(str(s)[:p]+format(e, f'0{q}b')[:q]+format(m, f'0{r}b')[:r], float_format):8.6e}|"
                ss += line + "\n"

    with open(f"output/{float_format.value}.md", "w") as f:
        f.write(ss)

    return

if __name__ == "__main__":
    test_converter()

    for float_format in FloatFormat:
        build_md(float_format)

    print("Finish")
