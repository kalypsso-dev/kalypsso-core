#!/usr/bin/env python

# -*- coding: utf-8 -*-

#  SPDX-FileCopyrightText: 2025 kalypsso contributors
#
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, getopt, os
import argparse
import math

from lxml import etree

import h5py

"""Helper tools for manipulating kalypsso's HDF5 output.

The original behavior of kalypsso was to offer the user the possibility to save both
simulation data and mesh data (node coordinates and node connectivity). Here we provide
the possibility to recreate the mesh data and xdmf from a kalypsso hdf5 output that
contains only the AMR orchard keys. This will make hdf5 output lighter as it avoids
saving mesh node coordinates and mesh node connectivity data in hdf5 outputs.
"""

class Hdf5Properties:
    """Utility class for reading kalypsso hdf5 output files properties."""

    def __init__(self, hdf5_name):
        self.h5file = h5py.File(hdf5_name)
        self.block_sizes = self.h5file["amr"].attrs["block_sizes"]
        self.dim = self.block_sizes.size
        self.keys = list(self.h5file.keys())

    def dim(self):
        return self.dim

    def time(self):
        return self.h5file["run"].attrs["time"]

    def iteration(self):
        return self.h5file["run"].attrs["iteration"]

    def num_leaves(self):
        return self.h5file["amr"]["keys"].size

    def num_cells_per_leaf(self):
        return math.prod(self.block_sizes)

    def num_nodes_per_leaf(self):
        return math.prod(self.block_sizes+1)

    def global_num_cells(self):
        return self.num_cells_per_leaf() * self.num_leaves()

    def connectivity_shape(self):
        return self.h5file["unstructured_mesh"]["connectivity"].shape

    def coordinates_shape(self):
        return self.h5file["unstructured_mesh"]["coordinates"].shape

    def are_mesh_data_available(self):
        for k in self.keys:
            if k == "unstructured_mesh":
                return True
        return False

def get_number_type_str(data_typename):
    if data_typename == "int32" or data_typename == "int":
        return "Int", 4
    elif data_typename == "uint32" or data_typename == "uint":
        return "UInt", 4
    elif data_typename == "uint64":
        return "UInt", 8
    elif data_typename == "float32" or data_typename == "float":
        return "Float", 4
    elif data_typename == "float64" or data_typename == "double":
        return "Float", 8
    else:
        return "Unknown", 0

def are_mesh_data_available(hdf5_name):
    """Return true if mesh data (coordinates and connectivity) are available in hdf5 file.
    """

    # check that hdf5 file actually exists
    if not os.path.exists(hdf5_name):
        print("File {} doesn't exist !".format(hdf5_name))
        return False
    else:
        h5prop = Hdf5Properties(hdf5_name)
        avail = h5prop.are_mesh_data_available()
        print("Checking if mesh data are available {} : {}".format(hdf5_name,avail))
        return avail

def create_xdmf(hdf5_name, mesh_data_are_in_separate_file):
    '''Given an kalypsso output as a hdf5 file, (re-)create associated xdmf file (for reading in ParaView).
    '''

    # check that hdf5 file actually exists
    if not os.path.exists(hdf5_name):
        print("File {} doesn't exist !".format(hdf5_name))
    else:
        print("Creating xdmf for {}".format(hdf5_name))

        h5prop = Hdf5Properties(hdf5_name)

        base_name = os.path.basename(os.path.splitext(hdf5_name)[0])
        xmf_name = base_name+'.xmf'

        xmf = etree.Element('Xdmf')
        xmf.set("Version", "2.0")
        domain = etree.SubElement(xmf, "Domain")

        grid = etree.SubElement(domain, "Grid")
        grid.set("Name", base_name)
        grid.set("GridType", "Uniform")

        time = etree.SubElement(grid, "Time")
        time.set("TimeType", "Single")
        time.set("Value", "{}".format(h5prop.time()))

        topology = etree.SubElement(grid, "Topology")
        topology.set("TopologyType", "Quadrilateral")
        topology.set("NumberOfElements", "{}".format(h5prop.global_num_cells()))

        dataitem = etree.SubElement(topology, "DataItem")
        dataitem.set("Dimensions", "{} {}".format(h5prop.connectivity_shape()[0], h5prop.connectivity_shape()[1]))
        dataitem.set("DataType", "Int")
        dataitem.set("Format", "HDF")
        dataitem.text = base_name+'.h5'+':/unstructured_mesh/connectivity'

        geometry = etree.SubElement(grid, "Geometry")
        geometry.set("GeometryType", "XYZ")
        dataitem_g = etree.SubElement(geometry, "DataItem")
        dataitem_g.set("Dimensions", "{} {}".format(h5prop.coordinates_shape()[0], h5prop.coordinates_shape()[1]))
        dataitem_g.set("NumberType", "Float")
        dataitem_g.set("Precision", "4")
        dataitem_g.set("Format", "HDF")
        dataitem_g.text = base_name+'.h5'+':/unstructured_mesh/coordinates'

        for item in h5prop.h5file["celldata"].keys():
            #print("{}".format(item))
            attr = etree.SubElement(grid, "Attribute")
            attr.set("Name", item)
            attr.set("AttributeType", "Scalar")
            attr.set("Center", "Cell")
            ditem = etree.SubElement(attr, "DataItem")
            data_typename = h5prop.h5file["celldata"][item].dtype.name
            number_type, precision = get_number_type_str(data_typename)
            ditem.set("NumberType", "{}".format(number_type))
            ditem.set("Precision", "{}".format(precision))
            ditem.set("Format", "HDF")
            ditem.set("Dimensions", "{}".format(h5prop.h5file["celldata"][item].shape[0]))
            ditem.text = base_name+'.h5'+':/celldata/'+item

        doc = etree.ElementTree(xmf)
        etree.indent(doc, space="    ")
        doc.write(file=xmf_name, xml_declaration=True,
                  encoding='UTF-8', doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')


def create_mesh_data(hdf5_name):
    # TODO
    return 0

###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create xdmf file for kalypsso hdf5 output.')
    parser.add_argument('--hdf5', type=str, default='test_blast_2d_iter0000000.h5', help='hdf5 file name')
    args = parser.parse_args()

    # should we create mesh data (in a separate hdf5 file)
    if are_mesh_data_available(args.hdf5):
        mesh_data_are_in_separate_file = False
        create_xdmf(args.hdf5, mesh_data_are_in_separate_file)
    else:
        # create mesh data
        #create_mesh_data(args.hdf5)
        mesh_data_are_in_separate_file = True
        create_xdmf(args.hdf5, mesh_data_are_in_separate_file)
