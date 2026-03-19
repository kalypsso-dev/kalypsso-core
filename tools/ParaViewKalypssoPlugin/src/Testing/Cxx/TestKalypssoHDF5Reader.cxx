// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vtkAutoInit.h>

// #include <vtkDataSet.h>
// #include <vtkDataSetWriter.h>
#include <vtkKalypssoHDF5Reader.h>
#include <vtkInformation.h>
#include <vtkNew.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkTrivialProducer.h>
#include <vtkXMLUniformGridAMRWriter.h>

#include <vtksys/SystemTools.hxx>
#include <vtksys/CommandLineArguments.hxx>

#include <iostream>

int
vtkIOKalypssoCxxTests(int argc, char ** argv)
{
  std::string filein;
  std::string varname;

  // double TimeStep = 4.2898e-05;

  vtksys::CommandLineArguments args;
  args.Initialize(argc, argv);
  args.AddArgument("-f",
                   vtksys::CommandLineArguments::SPACE_ARGUMENT,
                   &filein,
                   "(names of the kalypsso (HDF5) files to read)");
  args.AddArgument("-var",
                   vtksys::CommandLineArguments::SPACE_ARGUMENT,
                   &varname,
                   "(the name of the SCALAR variable to display)");

  if (!args.Parse() || argc == 1 || filein.empty())
  {
    std::cerr << "\nTestKalypssoHDF5Reader: \n"
              << "options are:\n";
    std::cerr << args.GetHelp() << "\n";
    return EXIT_FAILURE;
  }

  if (!vtksys::SystemTools::FileExists(filein.c_str()))
  {
    std::cerr << "\nFile " << filein.c_str() << " does not exist\n\n";
    return EXIT_FAILURE;
  }

  vtkNew<vtkKalypssoHDF5Reader> reader;
  reader->DebugOff();
  reader->SetFileName(filein.c_str());
  reader->UpdateInformation();

  for (auto i = 0; i < reader->GetNumberOfCellArrays(); i++)
    std::cout << "found array (" << i << ") = " << reader->GetCellArrayName(i) << "\n";
  reader->DisableAllCellArrays();
  reader->SetCellArrayStatus("rho", 1);
  reader->Update();

  // just for cross-check
  {
    vtkNew<vtkTrivialProducer> producer;
    producer->SetOutput(reader->GetOutput());

    std::string                        filename = "./test_NonOverlappingAMR_2d.vth";
    vtkNew<vtkXMLUniformGridAMRWriter> writer;
    writer->SetInputConnection(producer->GetOutputPort());
    writer->SetFileName(filename.c_str());
    writer->Write();
  }

  // reader->UpdateTimeStep(TimeStep); // time value
  // reader->Update();

  // double range[2];

  // if (varname.size())
  // {
  //   reader->GetOutput()->GetPointData()->GetArray(0)->GetRange(range);
  //   std::cerr << varname.c_str() << ": scalar range = [" << range[0] << ", " << range[1] <<
  //   "]\n";
  // }


  std::cout << *reader;
  /*
  VTK_CREATE(vtkDataSetWriter, writer);
  writer->SetInputData(FirstBlock);
  writer->SetFileTypeToBinary();
  writer->SetFileName("/tmp/foo.vtk");
  writer->Write();
  */

  return EXIT_SUCCESS;
}

// =================================================================
// =================================================================
int
main(int argc, char ** argv)
{
  const auto status = vtkIOKalypssoCxxTests(argc, argv);
  return status;
}
