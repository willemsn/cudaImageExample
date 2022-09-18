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

#include "ArgsImgTest.h"

ArgsImgTest::ArgsImgTest()
  : verbose(false),
    numCPUs(1), numGPUs(1),
    writeToOneOutput(false),
    inputFileName(""), outputFileName("")
{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
  reg("inputfile", "input file name to use", ArgumentParsing::STRING, 'i');
  reg("outputfile", "output file name to use", ArgumentParsing::STRING, 'o');
  reg("numcpus", "num of cores to use", ArgumentParsing::INT, 'n');
  reg("numgpus", "num of GPU devices to use", ArgumentParsing::INT, 'g');
  reg("writeMode", "Write to One Output", ArgumentParsing::NONE, 'w');
}

void ArgsImgTest::process(int argc, char *argv[])
{
  processCommandLineArgs(argc, argv);

  if (isSet("help"))
    {
      printUsage();
      exit(EXIT_SUCCESS);
    }

  verbose = isSet("verbose");
  if (verbose) { std::cout << "Verbose Output: ON" << std::endl; }
  
  writeToOneOutput = isSet("writeMode");
  if (verbose) { std::cout << "Writing to One Output: ON" << std::endl; }

  isSet("numcpus", numCPUs);
  if (verbose) { std::cout << "Setting num CPUs to " << numCPUs << std::endl; }

  isSet("numgpus", numGPUs);
  if (verbose) { std::cout << "Setting num GPUs to " << numGPUs << std::endl; }

  isSet("inputfile", inputFileName);
  if (verbose) { std::cout << "Setting inputFileName to " << inputFileName << std::endl; }
  
  isSet("outputfile", outputFileName);
  if (verbose) { std::cout << "Setting outputFileName to " << outputFileName << std::endl; }
}

