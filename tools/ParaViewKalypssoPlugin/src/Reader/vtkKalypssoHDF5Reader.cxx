// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define PARALLEL_DEBUG

#include "vtkKalypssoHDF5Reader.h"

#include <vtkAMRBox.h>
#include <vtkCallbackCommand.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkDataArraySelection.h>
#include <vtkDataObjectTypes.h>
#include <vtkDataSetAttributes.h>
#include <vtkErrorCode.h>
#include <vtkFieldData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkIntArray.h>
#include <vtkObjectFactory.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkTimerLog.h>
#include <vtkUniformGrid.h>

#include <sys/time.h>
#include <algorithm>
#include <vector>
#include <string>
#include <hdf5.h>

#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#include <vtkMPI.h>
#include <set>
#include <sstream>


vtkStandardNewMacro(vtkKalypssoHDF5Reader);

// =========================================================================
// =========================================================================
// Computes the cell center for the cell corresponding to cellIdx w.r.t.
// the given grid. The cell center is stored in the supplied buffer c.
std::array<double, 3>
ComputeCellCenter(vtkUniformGrid * grid, const int cellIdx)
{
  assert("pre: grid != NULL" && (grid != nullptr));
  assert("pre: cellIdx in bounds" && (cellIdx >= 0) && (cellIdx < grid->GetNumberOfCells()));

  vtkCell * myCell = grid->GetCell(cellIdx);
  assert("post: cell is NULL" && (myCell != nullptr));

  double   pCenter[3];
  double * weights = new double[myCell->GetNumberOfPoints()];
  int      subId = myCell->GetParametricCenter(pCenter);

  std::array<double, 3> c;
  myCell->EvaluateLocation(subId, pCenter, c.data(), weights);
  delete[] weights;
  return c;
}

// =========================================================================
// =========================================================================
void
attach_dummy_data_to_grid(vtkUniformGrid * grid, int dim, int block_size)
{

  vtkDoubleArray * user_data = vtkDoubleArray::New();
  user_data->SetName("user_data");
  user_data->SetNumberOfComponents(1);
  auto nbTuples = dim == 2 ? block_size * block_size : block_size * block_size * block_size;
  user_data->SetNumberOfTuples(nbTuples);

  for (int cellIdx = 0; cellIdx < grid->GetNumberOfCells(); ++cellIdx)
  {
    auto center = ComputeCellCenter(grid, cellIdx);

    const auto value = center[0] + center[1] + center[2];

    user_data->SetTuple1(cellIdx, value);
  }

  grid->GetCellData()->AddArray(user_data);
  user_data->Delete();
}

//
// PUBLIC MEMBERS
//

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::PrintSelf(ostream & os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "FileName: " << (this->FileName ? this->FileName : "(NULL)") << std::endl;
  os << indent << "dim: " << this->dim << std::endl;
  os << indent << "level_min: " << this->level_min << std::endl;
  os << indent << "level_max: " << this->level_max << std::endl;
  os << indent << "max level to read: " << this->MaximumLevelsToReadByDefault << std::endl;

  os << indent << "level_histogram:";
  for (auto num_level : this->level_histogram)
    os << " " << num_level;
  os << std::endl;

  os << indent << "block_sizes:";
  for (auto size : this->block_sizes)
    os << " " << size;
  os << std::endl;

  os << indent << "brick_sizes:";
  for (auto size : this->brick_sizes)
    os << " " << size;
  os << std::endl;

  os << indent << "scaling_factor: " << this->scaling_factor << std::endl;

  os << indent << "xyz_min:";
  for (auto coord : this->xyz_min)
    os << " " << coord;
  os << std::endl;
}

// ================================================================================
// ================================================================================
int
vtkKalypssoHDF5Reader::GetNumberOfCellArrays()
{
  return this->CellDataArraySelection->GetNumberOfArrays();
}

// ================================================================================
// ================================================================================
const char *
vtkKalypssoHDF5Reader::GetCellArrayName(int index)
{
  return this->CellDataArraySelection->GetArrayName(index);
}

// ================================================================================
// ================================================================================
int
vtkKalypssoHDF5Reader::GetCellArrayStatus(const char * name)
{
  return this->CellDataArraySelection->ArrayIsEnabled(name);
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::SetCellArrayStatus(const char * name, int status)
{
  if (status)
  {
    this->CellDataArraySelection->EnableArray(name);
  }
  else
  {
    this->CellDataArraySelection->DisableArray(name);
  }
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::SelectionModifiedCallback(vtkObject *,
                                                 unsigned long,
                                                 void * clientdata,
                                                 void *)
{
  static_cast<vtkKalypssoHDF5Reader *>(clientdata)->Modified();
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::EnableCellArray(const char * name)
{
  this->SetCellArrayStatus(name, 1);
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::DisableCellArray(const char * name)
{
  this->SetCellArrayStatus(name, 0);
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::EnableAllCellArrays()
{
  this->CellDataArraySelection->EnableAllArrays();
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::DisableAllCellArrays()
{
  this->CellDataArraySelection->DisableAllArrays();
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::SetFileName(const char * fileName)
{
  if (this->FileName != nullptr)
  {
    delete[] this->FileName;
    this->FileName = nullptr;
  }
  this->FileName = new char[strlen(fileName) + 1];
  strcpy(this->FileName, fileName);
  this->FileName[strlen(fileName)] = '\0';

  this->ReadMetaData();
  this->SetUpDataArraySelections();
  this->InitializeArraySelections();

  // this->Modified();
}

//
// PROTECTED MEMBERS
//

// ================================================================================
// ================================================================================
vtkKalypssoHDF5Reader::vtkKalypssoHDF5Reader()
{
  this->InitialRequest = true;
  this->FileName = nullptr;
  this->MetaDataLoaded = false;
  this->DebugOff();
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);
  this->CellDataArraySelection = vtkDataArraySelection::New();
  this->SelectionObserver = vtkCallbackCommand::New();
  this->SelectionObserver->SetCallback(&vtkKalypssoHDF5Reader::SelectionModifiedCallback);
  this->SelectionObserver->SetClientData(this);
  this->CellDataArraySelection->AddObserver(vtkCommand::ModifiedEvent, this->SelectionObserver);
  this->NumberOfTimeSteps = 0;
  this->TimeStep = 0;
  this->ActualTimeStep = 0;
  this->TimeStepTolerance = 1E-6;

  // all these will be populated in RequestInformation
  this->dim = 0;
  this->level_min = 0;
  this->level_max = 0;

  this->scaling_factor = 1.0;
  this->xyz_min.assign(3, 0.0);

  this->MaximumLevelsToReadByDefault = 1;

  // default values (should be initialized by reading hdf5 file)
  this->varnames = {};
}

// ================================================================================
// ================================================================================
vtkKalypssoHDF5Reader::~vtkKalypssoHDF5Reader()
{
  vtkDebugMacro(<< "cleaning up inside destructor");
  if (this->FileName)
    delete[] this->FileName;

  this->CellDataArraySelection->RemoveObserver(this->SelectionObserver);
  this->CellDataArraySelection->Delete();
  this->CellDataArraySelection = nullptr;
  this->SelectionObserver->Delete();
}

// ================================================================================
// ================================================================================
int
vtkKalypssoHDF5Reader::CanReadFile(const char * fname)
{
  bool valid = true;
  try
  {
    // check that file can be open, and that group amr and celldata exist
    HighFive::File file(std::string(this->FileName), HighFive::File::ReadOnly);
    auto           amr = file.getGroup("/amr");
    auto           celldata = file.getGroup("/celldata");
  }
  catch (const std::exception & e)
  {
    std::cerr << e.what() << std::endl;
    valid = false;
  }
  return valid;
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::ReadMetaData()
{
  HighFive::File file(std::string(this->FileName), HighFive::File::ReadOnly);

  auto celldata_group = file.getGroup("/celldata");

  // get a list of available cell data
  this->varnames = celldata_group.listObjectNames();
  // for (auto const & var_name : this->varnames)
  //   std::cout << var_name << "\n";
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::SetUpDataArraySelections()
{
  for (auto warn : this->varnames)
    this->CellDataArraySelection->AddArray(warn.c_str());
}

// ================================================================================
// ================================================================================
void
vtkKalypssoHDF5Reader::InitializeArraySelections()
{
  if (this->InitialRequest)
  {
    this->CellDataArraySelection->DisableAllArrays();
    this->InitialRequest = false;
  }
}
// ================================================================================
// ================================================================================
int
vtkKalypssoHDF5Reader::RequestInformation(vtkInformation *        request,
                                          vtkInformationVector ** inputVector,
                                          vtkInformationVector *  outputVector)
{
  if (this->MetaDataLoaded)
  {
    return (1);
  }

  this->Superclass::RequestInformation(request, inputVector, outputVector);

  vtkDebugMacro(<< "vtkKalypssoHDF5Reader::RequestInformation: Parsing file " << this->FileName
                << " for fields.");

  HighFive::File file(std::string(this->FileName), HighFive::File::ReadOnly);

  vtkInformation * outInfo = outputVector->GetInformationObject(0);
  outInfo->Set(CAN_HANDLE_PIECE_REQUEST(), 1);

  // amr_output is allocated in RequestDataObject
  vtkDataObject *        doOutput = outInfo->Get(vtkDataObject::DATA_OBJECT());
  vtkNonOverlappingAMR * amr_output = vtkNonOverlappingAMR::SafeDownCast(doOutput);

  // don't know yet if that is useful for us
  // outInfo->Set(vtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);

  auto amr = file.getGroup("/amr");

  // get dimension
  amr.getAttribute("dim").read(this->dim);

  // get level_min / level_max
  amr.getAttribute("level_min").read(this->level_min);
  amr.getAttribute("level_max").read(this->level_max);

  // get number of blocks per levels (level histogram is a HDF5 attribute in group "/amr")
  amr.getAttribute("level_histogram").read(this->level_histogram);

  // get block and brick sizes
  amr.getAttribute("block_sizes").read(this->block_sizes);
  amr.getAttribute("brick_sizes").read(this->brick_sizes);

  amr.getAttribute("scaling_factor").read(this->scaling_factor);
  amr.getAttribute("xyz_min").read(this->xyz_min);

  HighFive::DataSet amr_keys_ds = file.getDataSet("/amr/keys");
  amr_keys_ds.read(this->amr_keys);

  HighFive::DataSet amr_level_indexes_ds = file.getDataSet("/amr/level_indexes");
  amr_level_indexes_ds.read(this->amr_level_indexes);

  const auto num_levels = this->level_max - this->level_min + 1;
  amr_output->Initialize(num_levels, this->level_histogram.data());

  if (this->dim == 2)
  {
    this->setup_geometry<2>(amr_output);
  }
  else if (this->dim == 3)
  {
    this->setup_geometry<3>(amr_output);
  }

  // see later if we want to handle multiple time steps
  this->NumberOfTimeSteps = 1;
  this->TimeStepValues.assign(this->NumberOfTimeSteps, 0.0);
  for (int i = 0; i < NumberOfTimeSteps; i++)
    this->TimeStepValues[i] = 0; // t_start + i * (dt);

  double timeRange[2];
  timeRange[0] = this->TimeStepValues.front();
  timeRange[1] = this->TimeStepValues.back();

  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), timeRange, 2);
  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(),
               this->TimeStepValues.data(),
               static_cast<int>(this->TimeStepValues.size()));

  this->MetaDataLoaded = true;

  return 1;
}

// ================================================================================
// ================================================================================
int
vtkKalypssoHDF5Reader::RequestDataObject(vtkInformation *        vtkNotUsed(request),
                                         vtkInformationVector ** vtkNotUsed(inputVector),
                                         vtkInformationVector *  outputVector)
{
  vtkDataObject * output = vtkDataObject::GetData(outputVector, 0);
  if (!output || !output->IsA("vtkNonOverlappingAMR"))
  {
    vtkDataObject * newDO = vtkDataObjectTypes::NewDataObject("vtkNonOverlappingAMR");
    if (newDO)
    {
      outputVector->GetInformationObject(0)->Set(vtkDataObject::DATA_OBJECT(), newDO);
      newDO->FastDelete();
      return 1;
    }
  }

  return 1;
}

// ================================================================================
// ================================================================================
int
vtkKalypssoHDF5Reader::RequestData(vtkInformation *        vtkNotUsed(request),
                                   vtkInformationVector ** vtkNotUsed(inputVector),
                                   vtkInformationVector *  outputVector)
{
  vtkTimerLog::MarkStartEvent("vtkKalypssoHDF5Reader::RequestData");

  vtkDebugMacro(<< "RequestData(BEGIN)");
  vtkInformation *       outInfo = outputVector->GetInformationObject(0);
  vtkDataObject *        doOutput = outInfo->Get(vtkDataObject::DATA_OBJECT());
  vtkNonOverlappingAMR * amr_output = vtkNonOverlappingAMR::SafeDownCast(doOutput);

  // populate vtkNonOverlappingAMR geometry information
  const auto num_levels = this->level_max - this->level_min + 1;
  amr_output->Initialize(num_levels, this->level_histogram.data());

  if (this->dim == 2)
  {
    this->setup_geometry<2>(amr_output);
  }
  else if (this->dim == 3)
  {
    this->setup_geometry<3>(amr_output);
  }


  // int piece = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER());
  // int numPieces = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES());

  this->UpdateProgress(0.0);

  // TODO: make sure handling multiple time steps is ok here
  this->ActualTimeStep = 0;
  double requestedTimeValue = 0.0;
  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
  {
    requestedTimeValue = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

    for (int i = 0; i < this->NumberOfTimeSteps; i++)
    {
      if (fabs(requestedTimeValue - this->TimeStepValues[i]) < this->TimeStepTolerance)
      {
        this->ActualTimeStep = i;
        break;
      }
    }
    amr_output->GetInformation()->Set(vtkDataObject::DATA_TIME_STEP(), requestedTimeValue);
  }
  std::cout << "requestedTimeValue " << requestedTimeValue << endl;
  std::cout << "this->ActualTimeStep " << this->ActualTimeStep << endl;

  if (!this->FileName)
  {
    vtkErrorMacro(<< "error reading header specified!");
    return 0;
  }

  // this is where we populated AMR "heavy" data
  if (this->dim == 2)
  {
    this->Load_Variables<2>(amr_output);
  }
  else if (dim == 3)
  {
    this->Load_Variables<3>(amr_output);
  }

  this->UpdateProgress(1.0);

  vtkTimerLog::MarkEndEvent("vtkKalypssoHDF5Reader::RequestData");

  return 1;
}

// ================================================================================
// ================================================================================
bool
vtkKalypssoHDF5Reader::Is_Variable_Enabled(const char * vname)
{
  return GetCellArrayStatus(vname);
}
