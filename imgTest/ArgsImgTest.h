/****************************************************************************
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of cudaImageExample
 *
 * cudaImageExample is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * cudaImageExample is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with cudaImageExample. If not, see
 * <https://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#pragma once

#include <iostream>
#include "ArgumentParsing.h"

/**
 * @class ArgsImgTest
 * @brief Handles different commandline options and arguments
 * and places the values into variables.
 *
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */
class ArgsImgTest : public ArgumentParsing
{
public:
  ArgsImgTest();
  ~ArgsImgTest() = default;

  /**
   * Takes in the commandline arguments and places
   * them into variables.
   *
   * @param argc Number of commandline options/arguments
   * @param argv Array of strings for arguments
   */
  void process(int argc, char *argv[]);

  bool verbose;

  int numCPUs;
  int numGPUs;
  bool writeToOneOutput;
  std::string inputFileName, outputFileName;

private:
};
