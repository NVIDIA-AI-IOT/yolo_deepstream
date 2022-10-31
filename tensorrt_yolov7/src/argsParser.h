
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#ifndef __COMMAND_LINE_H_
#define __COMMAND_LINE_H_

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include <string.h>
#include <strings.h>

/**
 * args line parser 
 */
class argsParser {
public:
    argsParser(const int argc, char** argv);

    /**
     * Parse Flag
     */
    bool ParseFlag(const std::string argName) const;

    /**
     * Parse String
     */
    const char* ParseString(const std::string  argName) const;
    // const char* ParseString2(const std::string argName, const char* defaultValue = NULL, bool allowOtherDelimiters = true) const;

    /**
     * Parse String list delimited by ","
     */
    std::vector<std::string> ParseStringList(std::string argName, const char delimiter = ',') const;

    /**
     * The argument count that the object was created with from main()
     */
    int argc;
    char** argv;
};

#endif
