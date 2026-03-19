// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file vtkKalypssoHDF5Reader
 *
 * vtkKalypssoHDF5Reader is a source object that reads a kalypsso HDF5 file
 * and builds a vtkNonOverlappingAMR object.
 *
 * This plugins is inspired by the following sources from ParaView sources:
 *
 * - the Plugins and Examples/Plugins subfolders
 * - class vtkAMRBaseReader
 * - class vtkXMLUniformGridAMRReader (for RequestDataObject)
 *
 */
#ifndef KALYPSSO_PARAVIEW_PLUGIN_VTKKALYPSSOHDF5READER_H
#define KALYPSSO_PARAVIEW_PLUGIN_VTKKALYPSSOHDF5READER_H

#include "KalypssoHDF5ReaderModule.h" // for export macro
#include <vtkNonOverlappingAMRAlgorithm.h>

#include <vtkAMRBox.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkNew.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGrid.h>

class vtkCallbackCommand;

// from kalypsso
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h> // for compute_cell_length

#include <highfive/highfive.hpp>

#include <vector>
#include <string>

class vtkDataArraySelection;

//! compute cell center (only use here for debugging when creating a simple scalar field)
std::array<double, 3>
ComputeCellCenter(vtkUniformGrid * grid, const int cellIdx);

//! attaching scalar data to a uniform grid (i.e. a block of our vtkNonUniformAMR)
void
attach_dummy_data_to_grid(vtkUniformGrid * grid, int dim, int block_size);

// ================================================================================
// ================================================================================
// ================================================================================
class KALYPSSOHDF5READER_EXPORT vtkKalypssoHDF5Reader : public vtkNonOverlappingAMRAlgorithm
{
public:
  static vtkKalypssoHDF5Reader *
  New();

  void
  PrintSelf(ostream & os, vtkIndent indent) override;

  vtkTypeMacro(vtkKalypssoHDF5Reader, vtkNonOverlappingAMRAlgorithm);

  ///@{
  /**
   * This reader supports demand-driven heavy data reading i.e. downstream
   * pipeline can request specific blocks from the AMR using
   * vtkCompositeDataPipeline::UPDATE_COMPOSITE_INDICES() key in
   * RequestUpdateExtent() pass. However, when down-stream doesn't provide any
   * specific keys, the default behavior can be setup to read at-most N levels
   * by default. The number of levels read can be set using this method.
   * Set this to 0 to imply no limit. Default is 0.
   */
  vtkSetMacro(MaximumLevelsToReadByDefault, unsigned int);
  vtkGetMacro(MaximumLevelsToReadByDefault, unsigned int);
  ///@}


  vtkGetObjectMacro(CellDataArraySelection, vtkDataArraySelection);

  //! The observer to modify this object when the array selection is modified.
  vtkCallbackCommand * SelectionObserver;

  // Callback registered with the SelectionObserver.
  static void
  SelectionModifiedCallback(vtkObject *   caller,
                            unsigned long eid,
                            void *        clientdata,
                            void *        calldata);

  int
  GetNumberOfCellArrays();

  const char *
  GetCellArrayName(int index);

  int
  GetCellArrayStatus(const char * name);

  void
  SetCellArrayStatus(const char * name, int status);

  void
  EnableAllCellArrays();

  void
  DisableAllCellArrays();

  void
  EnableCellArray(const char * name);

  void
  DisableCellArray(const char * name);

  vtkGetFilePathMacro(FileName);
  virtual void
  SetFileName(VTK_FILEPATH const char * fileName);

protected:
  vtkKalypssoHDF5Reader();
  ~vtkKalypssoHDF5Reader();

  //! a simple test to see if hdf5 file exists, and is "valid" (i.e. contains a HDF group named
  //! "amr" at root level).
  int
  CanReadFile(VTK_FILEPATH const char * fname);

  void
  ReadMetaData();

  void
  SetUpDataArraySelections();

  void
  InitializeArraySelections();

  int
  RequestInformation(vtkInformation *        request,
                     vtkInformationVector ** inputVector,
                     vtkInformationVector *  outputVector);

  //! allocate a vtkNonOverlappingAMR
  int
  RequestDataObject(vtkInformation *        request,
                    vtkInformationVector ** inputVector,
                    vtkInformationVector *  outputVector);

  int
  RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

  bool   InitialRequest;
  char * FileName;

  bool MetaDataLoaded;

  vtkDataArraySelection * CellDataArraySelection;

  int              dim;
  int              level_min;
  int              level_max;
  std::vector<int> level_histogram;

  std::vector<int> block_sizes;
  std::vector<int> brick_sizes;

  double              scaling_factor;
  std::vector<double> xyz_min;

  std::vector<uint64_t> amr_keys;
  std::vector<uint64_t> amr_level_indexes;

  unsigned int MaximumLevelsToReadByDefault;

  bool
  Is_Variable_Enabled(const char * vname);

  template <int DIM>
  void
  Load_Variables(vtkNonOverlappingAMR * output);

private:
  vtkKalypssoHDF5Reader(const vtkKalypssoHDF5Reader &) = delete;

  template <int DIM>
  void
  setup_geometry(vtkNonOverlappingAMR * amr);

  void
  operator=(const vtkKalypssoHDF5Reader &) = delete;

  std::vector<std::string> varnames;
  std::vector<double>      TimeStepValues;
  int                      NumberOfTimeSteps;
  int                      TimeStep;
  int                      ActualTimeStep;
  double                   TimeStepTolerance;
}; // class vtkKalypssoHDF5Reader

// =================================================================
// =================================================================
template <int DIM>
void
vtkKalypssoHDF5Reader::setup_geometry(vtkNonOverlappingAMR * amr)
{

  Kokkos::Array<double, DIM> xyz_min_k;
  xyz_min_k[kalypsso::IX] = this->xyz_min[0];
  xyz_min_k[kalypsso::IY] = this->xyz_min[1];
  if constexpr (DIM == 3)
  {
    xyz_min_k[kalypsso::IZ] = this->xyz_min[2];
  }

  for (size_t i = 0; i < this->amr_keys.size(); ++i)
  {
    const auto key = this->amr_keys[i];
    const auto level = kalypsso::orchard_key_t<DIM>::level(key);

    if (level - this->level_min <= this->MaximumLevelsToReadByDefault)
    {

      // compute physical x,y,z for the block origin (lower left corner)
      constexpr auto centering = false;
      const auto     xyz_vertex = kalypsso::orchard_key_to_vertex_coord<DIM>(key, centering);
      const auto     xyz =
        kalypsso::vertex_coord_to_real_space<DIM>(xyz_vertex, this->scaling_factor, xyz_min_k);

      // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
      const auto dx = kalypsso::compute_cell_length<DIM>(level, this->block_sizes[kalypsso::IX]) *
                      this->scaling_factor;

      std::array<double, 3> origin{ xyz[kalypsso::IX], xyz[kalypsso::IY], 0.0 };
      std::array<double, 3> spacing{ dx, dx, 0.0 };
      if constexpr (DIM == 3)
      {
        origin[2] = xyz[kalypsso::IZ];
        spacing[2] = dx;
      }

      std::array<int, 3> lo{ 0, 0, 0 };
      std::array<int, 3> hi{ this->block_sizes[kalypsso::IX] - 1,
                             this->block_sizes[kalypsso::IY] - 1,
                             0 };
      if constexpr (DIM == 3)
      {
        hi[2] = this->block_sizes[kalypsso::IZ] - 1;
      }

      vtkAMRBox              box(lo.data(), hi.data());
      vtkNew<vtkUniformGrid> ug;
      ug->Initialize(&box, origin.data(), spacing.data());
      // attach_dummy_data_to_grid(ug, DIM, this->block_sizes[kalypsso::IX]);
      amr->SetDataSet(level - this->level_min, this->amr_level_indexes[i], ug);
    }
  }
} // vtkKalypssoHDF5Reader::setup_geometry

// ================================================================================
// ================================================================================
template <int DIM>
void
vtkKalypssoHDF5Reader::Load_Variables(vtkNonOverlappingAMR * amr_output)
{

  size_t numcells_per_block = this->block_sizes[kalypsso::IX] * this->block_sizes[kalypsso::IY];
  if constexpr (DIM == 3)
  {
    numcells_per_block *= this->block_sizes[kalypsso::IZ];
  }

  Kokkos::Array<double, DIM> xyz_min_k;
  xyz_min_k[kalypsso::IX] = this->xyz_min[0];
  xyz_min_k[kalypsso::IY] = this->xyz_min[1];
  if constexpr (DIM == 3)
  {
    xyz_min_k[kalypsso::IZ] = this->xyz_min[2];
  }

  // open hdf5 file
  HighFive::File file(std::string(this->FileName), HighFive::File::ReadOnly);

  // populate AMR structure block by block
  // loop over all AMR blocks (identified by an orchard key)
  for (size_t i = 0; i < this->amr_keys.size(); ++i)
  {
    const auto key = this->amr_keys[i];
    const auto level = kalypsso::orchard_key_t<DIM>::level(key);

    if (level - this->level_min <= this->MaximumLevelsToReadByDefault)
    {
      // compute physical x,y,z for the block origin (lower left corner)
      constexpr auto centering = false;
      const auto     xyz_vertex = kalypsso::orchard_key_to_vertex_coord<DIM>(key, centering);
      const auto     xyz =
        kalypsso::vertex_coord_to_real_space<DIM>(xyz_vertex, this->scaling_factor, xyz_min_k);

      // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
      const auto dx = kalypsso::compute_cell_length<DIM>(level, this->block_sizes[kalypsso::IX]) *
                      this->scaling_factor;

      std::array<double, 3> origin{ xyz[kalypsso::IX], xyz[kalypsso::IY], 0.0 };
      std::array<double, 3> spacing{ dx, dx, 0.0 };
      if constexpr (DIM == 3)
      {
        origin[2] = xyz[kalypsso::IZ];
        spacing[2] = dx;
      }

      std::array<int, 3> lo{ 0, 0, 0 };
      std::array<int, 3> hi{ this->block_sizes[kalypsso::IX] - 1,
                             this->block_sizes[kalypsso::IY] - 1,
                             0 };
      if constexpr (DIM == 3)
      {
        hi[2] = this->block_sizes[kalypsso::IZ] - 1;
      }

      vtkAMRBox box(lo.data(), hi.data());

      // create a unoform grid that will contain all data fields
      vtkNew<vtkUniformGrid> ug;
      ug->Initialize(&box, origin.data(), spacing.data());

      // loop over all enabled variables, read data as needed
      for (size_t ivar = 0; ivar < this->varnames.size(); ivar++)
      {
        const char * vname = this->varnames[ivar].c_str();

        if (this->Is_Variable_Enabled(vname)) // is variable enabled in the ParaView GUI
        {

          auto              dataset_name = std::string("/celldata/") + this->varnames[ivar];
          HighFive::DataSet dataset = file.getDataSet(dataset_name);

          std::vector<size_t> offset{ i * numcells_per_block };
          std::vector<size_t> size{ numcells_per_block };
          HighFive::Selection dataset_slice = dataset.select(offset, size);

          std::vector<kalypsso::real_t> data;
          dataset_slice.read(data);

          {
            vtkDoubleArray * user_data = vtkDoubleArray::New();
            user_data->SetName(vname);
            user_data->SetNumberOfComponents(1);
            user_data->SetNumberOfTuples(numcells_per_block);

            for (int cellIdx = 0; cellIdx < ug->GetNumberOfCells(); ++cellIdx)
            {
              user_data->SetTuple1(cellIdx, data[cellIdx]);
            }

            ug->GetCellData()->AddArray(user_data);
            user_data->Delete();
          }
        }
      } // end for ivar

      amr_output->SetDataSet(level - this->level_min, this->amr_level_indexes[i], ug);
    }
  } // end for AMR keys/blocks

} // vtkKalypssoHDF5Reader::Load_Variables

#endif // KALYPSSO_PARAVIEW_PLUGIN_VTKKALYPSSOHDF5READER_H
