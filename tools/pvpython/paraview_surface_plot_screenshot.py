#
# Note: this script aimed to be run by pvbatch
#

# -*- coding: utf-8 -*-

import sys
import getopt
import os
import argparse

#### import the simple module from the paraview
from paraview.simple import *

def dump_screenshot(xdmf_filename, fieldname):

    # trace generated using paraview version 5.12.0
    #import paraview
    #paraview.compatibility.major = 5
    #paraview.compatibility.minor = 12

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'XDMF Reader'
    xmf_data = XDMFReader(registrationName='machin', FileNames=[xdmf_filename])

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # Properties modified on xmf_data
    xmf_data.CellArrayStatus = [fieldname]

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    xmf_data_display = Show(xmf_data, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    xmf_data_display.Representation = 'Surface'

    # reset view to fit data
    renderView1.ResetCamera(False, 0.9)

    #changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.5, 0.5, 3.35]
    renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    xmf_data_display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get color transfer function/color map for fieldname
    field_LUT = GetColorTransferFunction(fieldname)

    # get opacity transfer function/opacity map for fieldname
    field_PWF = GetOpacityTransferFunction(fieldname)

    # get 2D transfer function for fieldname
    field_TF2D = GetTransferFunction2D(fieldname)

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # hide color bar/color legend
    xmf_data_display.SetScalarBarVisibility(renderView1, False)

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1570, 1054)

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.5, 0.5, 3.35]
    renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
    renderView1.CameraParallelScale = 0.7071067811865476

    # save screenshot
    base_name, _ = os.path.splitext(xdmf_filename)
    png_filename = base_name + ".png"
    SaveScreenshot(filename=png_filename, viewOrLayout=renderView1, location=16, ImageResolution=[1570, 1054], TransparentBackground=1)

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(1570, 1054)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.5, 0.5, 3.35]
    renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
    renderView1.CameraParallelScale = 0.7071067811865476


###############################################################################
if __name__ == "__main__":
    '''Dump a surface plot into a PNG image.

    Example of use:
       pvbatch paraview_surface_plot_screenshot.py --filename data.xmf --field rho
    '''


    parser = argparse.ArgumentParser(description='Save surface plot into PNG image file.')
    parser.add_argument('--filename', type=str, default='data.xmf', help='kalypsso output data file in XDFM+HDF5 format')
    parser.add_argument('--field', type=str, default='rho', help='field name to display (e.g. rho)')
    args = parser.parse_args()

    dump_screenshot(args.filename, args.field)
