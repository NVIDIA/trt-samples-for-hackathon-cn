/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php
// https://github.com/rogersce/cnpy

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <typeinfo>
#include <vector>
#include <zlib.h>

namespace cnpy
{
struct NpyArray
{
    NpyArray(const std::vector<size_t> &_shape, size_t _word_size, bool _fortran_order):
        shape(_shape), word_size(_word_size), fortran_order(_fortran_order)
    {
        num_vals = 1;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            num_vals *= shape[i];
        }
        data_holder = std::shared_ptr<std::vector<char>>(new std::vector<char>(num_vals * word_size));
    }

    NpyArray():
        shape(0), word_size(0), fortran_order(0), num_vals(0) {}

    template<typename T>
    T *data()
    {
        return reinterpret_cast<T *>(&(*data_holder)[0]);
    }

    template<typename T>
    const T *data() const
    {
        return reinterpret_cast<T *>(&(*data_holder)[0]);
    }

    template<typename T>
    std::vector<T> as_vec() const
    {
        const T *p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }

    size_t num_bytes() const
    {
        return data_holder->size();
    }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t>                shape;
    size_t                             word_size;
    bool                               fortran_order;
    size_t                             num_vals;
};

using npz_t = std::map<std::string, NpyArray>;

char BigEndianTest()
{
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

char map_type(const std::type_info &t)
{
    if (t == typeid(float))
        return 'f';
    if (t == typeid(double))
        return 'f';
    if (t == typeid(long double))
        return 'f';

    if (t == typeid(int))
        return 'i';
    if (t == typeid(char))
        return 'i';
    if (t == typeid(short))
        return 'i';
    if (t == typeid(long))
        return 'i';
    if (t == typeid(long long))
        return 'i';

    if (t == typeid(unsigned char))
        return 'u';
    if (t == typeid(unsigned short))
        return 'u';
    if (t == typeid(unsigned long))
        return 'u';
    if (t == typeid(unsigned long long))
        return 'u';
    if (t == typeid(unsigned int))
        return 'u';

    if (t == typeid(bool))
        return 'b';

    if (t == typeid(std::complex<float>))
        return 'c';
    if (t == typeid(std::complex<double>))
        return 'c';
    if (t == typeid(std::complex<long double>))
        return 'c';

    else
        return '?';
}

template<typename T>
std::vector<char> &operator+=(std::vector<char> &lhs, const T rhs)
{
    //write in little endian
    for (size_t byte = 0; byte < sizeof(T); byte++)
    {
        char val = *((char *)&rhs + byte);
        lhs.push_back(val);
    }
    return lhs;
}

template<>
std::vector<char> &operator+=(std::vector<char> &lhs, const std::string rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template<>
std::vector<char> &operator+=(std::vector<char> &lhs, const char *rhs)
{
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; ++byte)
    {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

template<typename T>
std::vector<char> create_npy_header(const std::vector<size_t> &shape)
{
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); ++i)
    {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1)
        dict += ",";
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += (char)0x93;
    header += "NUMPY";
    header += (char)0x01; //major version of numpy format
    header += (char)0x00; //minor version of numpy format
    header += (uint16_t)dict.size();
    header.insert(header.end(), dict.begin(), dict.end());

    return header;
}

void parse_npy_header(unsigned char *buffer, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    //std::string magic_string(buffer,6);
    uint8_t     major_version = *reinterpret_cast<uint8_t *>(buffer + 6);
    uint8_t     minor_version = *reinterpret_cast<uint8_t *>(buffer + 7);
    uint16_t    header_len    = *reinterpret_cast<uint16_t *>(buffer + 8);
    std::string header(reinterpret_cast<char *>(buffer + 9), header_len);

    size_t loc1, loc2;

    //fortran order
    loc1          = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex  num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex))
    {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1              = header.find("descr") + 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    //char type = header[loc1+1];
    //assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2               = str_ws.find("'");
    word_size          = atoi(str_ws.substr(0, loc2).c_str());
}

void parse_npy_header(FILE *fp, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    char   buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    size_t loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex  num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex))
    {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    //char type = header[loc1+1];
    //assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2               = str_ws.find("'");
    word_size          = atoi(str_ws.substr(0, loc2).c_str());
}

void parse_zip_footer(FILE *fp, uint16_t &nrecs, size_t &global_header_size, size_t &global_header_offset)
{
    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if (res != 22)
        throw std::runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no              = *(uint16_t *)&footer[4];
    disk_start           = *(uint16_t *)&footer[6];
    nrecs_on_disk        = *(uint16_t *)&footer[8];
    nrecs                = *(uint16_t *)&footer[10];
    global_header_size   = *(uint32_t *)&footer[12];
    global_header_offset = *(uint32_t *)&footer[16];
    comment_len          = *(uint16_t *)&footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}

NpyArray load_the_npy_file(FILE *fp)
{
    std::vector<size_t> shape;
    size_t              word_size;
    bool                fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    NpyArray arr(shape, word_size, fortran_order);
    size_t   nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

NpyArray load_the_npz_array(FILE *fp, uint32_t compr_bytes, uint32_t uncompr_bytes)
{
    std::vector<unsigned char> buffer_compr(compr_bytes);
    std::vector<unsigned char> buffer_uncompr(uncompr_bytes);
    size_t                     nread = fread(&buffer_compr[0], 1, compr_bytes, fp);
    if (nread != compr_bytes)
        throw std::runtime_error("load_the_npy_file: failed fread");

    int      err;
    z_stream d_stream;

    d_stream.zalloc   = Z_NULL;
    d_stream.zfree    = Z_NULL;
    d_stream.opaque   = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in  = Z_NULL;
    err               = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in  = compr_bytes;
    d_stream.next_in   = &buffer_compr[0];
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out  = &buffer_uncompr[0];

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);

    std::vector<size_t> shape;
    size_t              word_size;
    bool                fortran_order;
    parse_npy_header(&buffer_uncompr[0], word_size, shape, fortran_order);

    NpyArray array(shape, word_size, fortran_order);

    size_t offset = uncompr_bytes - array.num_bytes();
    memcpy(array.data<unsigned char>(), &buffer_uncompr[0] + offset, array.num_bytes());

    return array;
}

npz_t npz_load(std::string fname)
{
    FILE *fp = fopen(fname.c_str(), "rb");

    if (!fp)
    {
        throw std::runtime_error("npz_load: Error! Unable to open file " + fname + "!");
    }

    npz_t arrays;

    while (1)
    {
        std::vector<char> local_header(30);
        size_t            headerres = fread(&local_header[0], sizeof(char), 30, fp);
        if (headerres != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04)
            break;

        //read in the variable name
        uint16_t    name_len = *(uint16_t *)&local_header[26];
        std::string varname(name_len, ' ');
        size_t      vname_res = fread(&varname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        //erase the lagging .npy
        varname.erase(varname.end() - 4, varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t *)&local_header[28];
        if (extra_field_len > 0)
        {
            std::vector<char> buff(extra_field_len);
            size_t            efield_res = fread(&buff[0], sizeof(char), extra_field_len, fp);
            if (efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method  = *reinterpret_cast<uint16_t *>(&local_header[0] + 8);
        uint32_t compr_bytes   = *reinterpret_cast<uint32_t *>(&local_header[0] + 18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t *>(&local_header[0] + 22);

        if (compr_method == 0)
        {
            arrays[varname] = load_the_npy_file(fp);
        }
        else
        {
            arrays[varname] = load_the_npz_array(fp, compr_bytes, uncompr_bytes);
        }
    }

    fclose(fp);
    return arrays;
}

NpyArray npz_load(std::string fname, std::string varname)
{
    FILE *fp = fopen(fname.c_str(), "rb");

    if (!fp)
        throw std::runtime_error("npz_load: Unable to open file " + fname);

    while (1)
    {
        std::vector<char> local_header(30);
        size_t            header_res = fread(&local_header[0], sizeof(char), 30, fp);
        if (header_res != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04)
            break;

        //read in the variable name
        uint16_t    name_len = *(uint16_t *)&local_header[26];
        std::string vname(name_len, ' ');
        size_t      vname_res = fread(&vname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end() - 4, vname.end()); //erase the lagging .npy

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t *)&local_header[28];
        fseek(fp, extra_field_len, SEEK_CUR); //skip past the extra field

        uint16_t compr_method  = *reinterpret_cast<uint16_t *>(&local_header[0] + 8);
        uint32_t compr_bytes   = *reinterpret_cast<uint32_t *>(&local_header[0] + 18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t *>(&local_header[0] + 22);

        if (vname == varname)
        {
            NpyArray array = (compr_method == 0) ? load_the_npy_file(fp) : load_the_npz_array(fp, compr_bytes, uncompr_bytes);
            fclose(fp);
            return array;
        }
        else
        {
            //skip past the data
            uint32_t size = *(uint32_t *)&local_header[22];
            fseek(fp, size, SEEK_CUR);
        }
    }

    fclose(fp);

    //if we get here, we haven't found the variable in the file
    throw std::runtime_error("npz_load: Variable name " + varname + " not found in " + fname);
}

NpyArray npy_load(std::string fname)
{
    FILE *fp = fopen(fname.c_str(), "rb");

    if (!fp)
        throw std::runtime_error("npy_load: Unable to open file " + fname);

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}

template<typename T>
void npy_save(std::string fname, const T *data, const std::vector<size_t> shape, std::string mode = "w")
{
    FILE               *fp = NULL;
    std::vector<size_t> true_data_shape; //if appending, the shape of existing + new data

    if (mode == "a")
        fp = fopen(fname.c_str(), "r+b");

    if (fp)
    {
        //file exists. we need to append to it. read the header, modify the array size
        size_t word_size;
        bool   fortran_order;
        parse_npy_header(fp, word_size, true_data_shape, fortran_order);
        assert(!fortran_order);

        if (word_size != sizeof(T))
        {
            std::cout << "libnpy error: " << fname << " has word size " << word_size << " but npy_save appending data sized " << sizeof(T) << "\n";
            assert(word_size == sizeof(T));
        }
        if (true_data_shape.size() != shape.size())
        {
            std::cout << "libnpy error: npy_save attempting to append misdimensioned data to " << fname << "\n";
            assert(true_data_shape.size() != shape.size());
        }

        for (size_t i = 1; i < shape.size(); ++i)
        {
            if (shape[i] != true_data_shape[i])
            {
                std::cout << "libnpy error: npy_save attempting to append misshaped data to " << fname << "\n";
                assert(shape[i] == true_data_shape[i]);
            }
        }
        true_data_shape[0] += shape[0];
    }
    else
    {
        fp              = fopen(fname.c_str(), "wb");
        true_data_shape = shape;
    }

    std::vector<char> header = create_npy_header<T>(true_data_shape);
    size_t            nels   = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    fseek(fp, 0, SEEK_SET);
    fwrite(&header[0], sizeof(char), header.size(), fp);
    fseek(fp, 0, SEEK_END);
    fwrite(data, sizeof(T), nels, fp);
    fclose(fp);
}

template<typename T>
void npz_save(std::string zipname, std::string fname, const T *data, const std::vector<size_t> &shape, std::string mode = "w")
{
    //first, append a .npy to the fname
    fname += ".npy";

    //now, on with the show
    FILE             *fp                   = NULL;
    uint16_t          nrecs                = 0;
    size_t            global_header_offset = 0;
    std::vector<char> global_header;

    if (mode == "a")
        fp = fopen(zipname.c_str(), "r+b");

    if (fp)
    {
        //zip file exists. we need to add a new npy file to it.
        //first read the footer. this gives us the offset and size of the global header
        //then read and store the global header.
        //below, we will write the the new data at the start of the global header then append the global header and footer below it
        size_t global_header_size;
        parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);
        fseek(fp, global_header_offset, SEEK_SET);
        global_header.resize(global_header_size);
        size_t res = fread(&global_header[0], sizeof(char), global_header_size, fp);
        if (res != global_header_size)
        {
            throw std::runtime_error("npz_save: header read error while adding to existing zip");
        }
        fseek(fp, global_header_offset, SEEK_SET);
    }
    else
    {
        fp = fopen(zipname.c_str(), "wb");
    }

    std::vector<char> npy_header = create_npy_header<T>(shape);

    size_t nels   = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    size_t nbytes = nels * sizeof(T) + npy_header.size();

    //get the CRC of the data to be added
    uint32_t crc = crc32(0L, (uint8_t *)&npy_header[0], npy_header.size());
    crc          = crc32(crc, (uint8_t *)data, nels * sizeof(T));

    //build the local header
    std::vector<char> local_header;
    local_header += "PK";                   //first part of sig
    local_header += (uint16_t)0x0403;       //second part of sig
    local_header += (uint16_t)20;           //min version to extract
    local_header += (uint16_t)0;            //general purpose bit flag
    local_header += (uint16_t)0;            //compression method
    local_header += (uint16_t)0;            //file last mod time
    local_header += (uint16_t)0;            //file last mod date
    local_header += (uint32_t)crc;          //crc
    local_header += (uint32_t)nbytes;       //compressed size
    local_header += (uint32_t)nbytes;       //uncompressed size
    local_header += (uint16_t)fname.size(); //fname length
    local_header += (uint16_t)0;            //extra field length
    local_header += fname;

    //build global header
    global_header += "PK";                           //first part of sig
    global_header += (uint16_t)0x0201;               //second part of sig
    global_header += (uint16_t)20;                   //version made by
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    global_header += (uint16_t)0;                    //file comment length
    global_header += (uint16_t)0;                    //disk number where file starts
    global_header += (uint16_t)0;                    //internal file attributes
    global_header += (uint32_t)0;                    //external file attributes
    global_header += (uint32_t)global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
    global_header += fname;

    //build footer
    std::vector<char> footer;
    footer += "PK";                                                            //first part of sig
    footer += (uint16_t)0x0605;                                                //second part of sig
    footer += (uint16_t)0;                                                     //number of this disk
    footer += (uint16_t)0;                                                     //disk where footer starts
    footer += (uint16_t)(nrecs + 1);                                           //number of records on this disk
    footer += (uint16_t)(nrecs + 1);                                           //total number of records
    footer += (uint32_t)global_header.size();                                  //nbytes of global headers
    footer += (uint32_t)(global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
    footer += (uint16_t)0;                                                     //zip file comment length

    //write everything
    fwrite(&local_header[0], sizeof(char), local_header.size(), fp);
    fwrite(&npy_header[0], sizeof(char), npy_header.size(), fp);
    fwrite(data, sizeof(T), nels, fp);
    fwrite(&global_header[0], sizeof(char), global_header.size(), fp);
    fwrite(&footer[0], sizeof(char), footer.size(), fp);
    fclose(fp);
}

template<typename T>
void npy_save(std::string fname, const std::vector<T> data, std::string mode = "w")
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npy_save(fname, &data[0], shape, mode);
}

template<typename T>
void npz_save(std::string zipname, std::string fname, const std::vector<T> data, std::string mode = "w")
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npz_save(zipname, fname, &data[0], shape, mode);
}

} // namespace cnpy

#endif
