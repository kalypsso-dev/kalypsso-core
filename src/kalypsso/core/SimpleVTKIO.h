// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SimpleVTKIO.h
 */
#ifndef KALYPSSO_CORE_SIMPLE_VTK_IO_H_
#define KALYPSSO_CORE_SIMPLE_VTK_IO_H_

#include <kalypsso/core/HydroParams.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost
#include <kalypsso/core/FieldMap.h>

// #include "kalypsso/core/bitpit_common.h"
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/core/io_utils.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>

namespace kalypsso
{

/**
 * Simple VTK IO routine (simple means Partitioned VTU, using ASCII).
 *
 * Here we assume DataArray size is the same as the number of AMR mesh octants.
 *
 * \param[in] forest a forest_t const reference
 * \param[in] filename a string specify the output filename suffix (e.g. nb of iter)
 * \param[in] data a Kokkos::View to the data to save
 * \param[in] fm a field map to access data
 * \param[in] names2index a map of names (of scalar field to save) to id (to fm)
 * \param[in] config_map a ConfigMap object to access input parameter file data (ini file format)
 */
template <size_t dim, typename device_t, typename model_t>
void
writeVTK(typename p4est::Wrapper<dim>::forest_t *   forest,
         typename p4est::Wrapper<dim>::geometry_t * geom,
         DataArrayLeaf<real_t, device_t>            data,
         const model_t &                            model,
         std::string                                filename)
{

  using p4est_t = typename p4est::Wrapper<dim>;

  using locidx_t = p4est::locidx_t;

  // copy data from device to host
  auto datah = Kokkos::create_mirror(data);

  // copy device data to host
  Kokkos::deep_copy(datah, data);

  // dimension : 2 or 3 ?
  // constexpr uint8_t dim = dim_;

  const auto & id2names = model.get_id2names_map();
  const auto & fm = model.get_fieldmap();

  // number of scalar fields
  int numScalarFields = id2names.size();

  // create p4est vtk context
  typename p4est_t::vtk_context_t * context;
  context = p4est_t::vtk_context_new(forest, filename.c_str());

  if (geom != nullptr)
    p4est_t::vtk_context_set_geom(context, geom);

  // we do not write point data (only cell data), so it is safe to set
  // continuous to true.
  // this will not save any space though since the default scale is < 1.
  p4est_t::vtk_context_set_continuous(context, 1);

  // write vtk header (vertex positions and quadrant-to-vertex mapping)
  context = p4est_t::vtk_write_header(context);

  // write meta data (tree, level, MPI rank)
  int write_tree = 1;
  int write_level = 1;
  int write_rank = 1;
  int wrap_rank = 0;

  // local number of quadrants/octants
  auto numOcts = forest->local_num_quadrants;

  sc_array_t **            vtkdata = (sc_array_t **)malloc(sizeof(sc_array_t *) * 5);
  std::vector<std::string> varNames;

  // prepare sc_array_t with scalar field from our Kokkos::View
  int id = 0;
  for (auto iter : id2names)
  {

    // get variables string name
    const auto varName = iter.second;

    // get variable id
    auto iVar = static_cast<typename model_t::VarId>(iter.first);

    // allocate an sc_array_t
    vtkdata[id] = sc_array_new_size(sizeof(double), numOcts);

    // register scalar field name
    varNames.push_back(varName);

    // copy data to vtkdata
    for (locidx_t iOct = 0; iOct < numOcts; iOct++)
    {

      double * ptr = (double *)sc_array_index(vtkdata[id], iOct);
      ptr[0] = datah(iOct, fm[iVar]);
    }

    ++id;

  } // end for iter

  // call p4est vtk API to write all scalar fields at once
  switch (numScalarFields)
  {
    case 1:
      context = p4est_t::vtk_write_cell_dataf(context,
                                              write_tree,
                                              write_level,
                                              write_rank,
                                              wrap_rank,
                                              numScalarFields, /*  nb scalars */
                                              0,               /*  nb vector fields */
                                              varNames[0].c_str(),
                                              vtkdata[0],
                                              context);
      break;
    case 2:
      context = p4est_t::vtk_write_cell_dataf(context,
                                              write_tree,
                                              write_level,
                                              write_rank,
                                              wrap_rank,
                                              numScalarFields, /*  nb scalars */
                                              0,               /*  nb vector fields */
                                              varNames[0].c_str(),
                                              vtkdata[0],
                                              varNames[1].c_str(),
                                              vtkdata[1],
                                              context);
      break;
    case 3:
      context = p4est_t::vtk_write_cell_dataf(context,
                                              write_tree,
                                              write_level,
                                              write_rank,
                                              wrap_rank,
                                              numScalarFields, /*  nb scalars */
                                              0,               /*  nb vector fields */
                                              varNames[0].c_str(),
                                              vtkdata[0],
                                              varNames[1].c_str(),
                                              vtkdata[1],
                                              varNames[2].c_str(),
                                              vtkdata[2],
                                              context);
      break;
    case 4:
      context = p4est_t::vtk_write_cell_dataf(context,
                                              write_tree,
                                              write_level,
                                              write_rank,
                                              wrap_rank,
                                              numScalarFields, /*  nb scalars */
                                              0,               /*  nb vector fields */
                                              varNames[0].c_str(),
                                              vtkdata[0],
                                              varNames[1].c_str(),
                                              vtkdata[1],
                                              varNames[2].c_str(),
                                              vtkdata[2],
                                              varNames[3].c_str(),
                                              vtkdata[3],
                                              context);
      break;
    case 5:
      context = p4est_t::vtk_write_cell_dataf(context,
                                              write_tree,
                                              write_level,
                                              write_rank,
                                              wrap_rank,
                                              numScalarFields, /*  nb scalars */
                                              0,               /*  nb vector fields */
                                              varNames[0].c_str(),
                                              vtkdata[0],
                                              varNames[1].c_str(),
                                              vtkdata[1],
                                              varNames[2].c_str(),
                                              vtkdata[2],
                                              varNames[3].c_str(),
                                              vtkdata[3],
                                              varNames[4].c_str(),
                                              vtkdata[4],
                                              context);
      break;
    default:
      fprintf(stderr,
              "Wrong number of scalar fields. You should consider refactoring writeVTK routine.\n");
      break;
  }

  // write vtk file footer
  p4est_t::vtk_write_footer(context);

  // destroy intermediate sc_array_t's
  id = 0;
  for (auto iter : id2names)
  {
    sc_array_destroy(vtkdata[id]);
    ++id;
  }

  free(vtkdata);

} // writeVTK

} // namespace kalypsso

#endif // KALYPSSO_CORE_SIMPLE_VTK_IO_H_
