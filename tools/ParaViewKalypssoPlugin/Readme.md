# Paraview plugin for loading kalypsso HDF5 data

## Motivations

The purpose of this Paraview plugin is to provide an alternative way of loading kalypsso data into paraview.

### Reducing file size

Currently kalypsso output uses XDMF+HDF5 file format, and ParaView loads data as a VTU object (VTK Unstructured Grid). Let's just compute the file size overhead to store the mesh in the HDF5 file. A given AMR mesh contains $N$ octants, $(n+1)^d$ points per octant, where $n$ is the block size and $d=2,3$ is dimension. This means, the mesh is represented by
- an array of points coordinates of size $3 N (n+1)^d$ using float
- an array of cell connectivity of size $2^d N n^d$ using integer
while each physics field (rho, momentum, ...) is represented by an array of doubles of size $N n^d$

For a pure 3D hydrodynamics simulation (5 fields), the mesh size overhead is roughly $11 Nn^3$ whereas the physics fields size is $5 N n^d$.

We could avoid dumping the mesh into the hdf5 file. Instead, we could just dump the orchard keys (1 per octant, so just an array of $N$ 64 bits integer), since the mesh point coordinates can be reconstructed from it.

### Use a VTK AMR data format

Current kalypsso xdmf+hdf5 output are loaded into paraview as an unstructured grid. It would be good to be able to use one of the three native AMR data formats available in VTK, namely:

- [vtkNonOverlappingAMR](https://vtk.org/doc/nightly/html/classvtkNonOverlappingAMR.html) : this class is perfectly adapted to kalypsso data structure
- [vtkOverlappingAMR](https://vtk.org/doc/nightly/html/classvtkOverlappingAMR.html): irrelevant here
- [vtkHyperTreeGrid](https://vtk.org/doc/nightly/html/classvtkHyperTreeGrid.html): requires some work to adapt kalypsso data structure to HTG.

We will first focus on designing a paraview plugin for loading kalypsso HDF5 file and fill a `vtkNonOverlappingAMR`. Later we will consider an upgrade of the plugin to data into a HTG.

### Resources

Some links to help writing a ParaView plugin:

- https://kitware.github.io/paraview-docs/nightly/cxx/PluginHowto.html
- https://www.paraview.org/Wiki/ParaView/Plugin_HowTo (this is deprecated)

Some examples by J. Favre:
- https://github.com/jfavre/ParaViewSalvusPlugin
- https://github.com/jfavre/ParaViewNek5000Plugin

### Additional notes

See if the ParaView plugin code could be mutualized with a potential catalyst plugin.
