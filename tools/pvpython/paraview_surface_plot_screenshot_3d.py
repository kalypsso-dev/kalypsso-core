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

    # create a new 'Xdmf3 Reader S'
    xmf_data = XDMFReader(registrationName='my_data', FileNames=[xdmf_filename])

    # Properties modified on xmf_data
    xmf_data.CellArrayStatus = [fieldname]

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    xmf_dataDisplay = Show(xmf_data, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    xmf_dataDisplay.Representation = 'Surface'

    # reset view to fit data
    renderView1.ResetCamera(False, 0.9)

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # show color bar/color legend
    xmf_dataDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get color transfer function/color map for 'fieldname'
    fieldnameLUT = GetColorTransferFunction(fieldname)

    # get color legend/bar for fieldnameLUT in view renderView1
    fieldnameLUTColorBar = GetScalarBar(fieldnameLUT, renderView1)

    # change scalar bar placement
    fieldnameLUTColorBar.Position = [0.8883429204621378, 0.8430895725424731]
    fieldnameLUTColorBar.ScalarBarLength = 0.3300000000000011

    # get opacity transfer function/opacity map for 'fieldname'
    fieldnamePWF = GetOpacityTransferFunction(fieldname)

    # get 2D transfer function for 'fieldname'
    fieldnameTF2D = GetTransferFunction2D(fieldname)

    # create a new 'Clip'
    clip1 = Clip(registrationName='Clip1', Input=xmf_data)

    # show data in view
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    clip1Display.Representation = 'Surface'

    # hide data in view
    Hide(xmf_data, renderView1)

    # show color bar/color legend
    clip1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # set active source
    SetActiveSource(xmf_data)

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=clip1.ClipType)

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1090, 995)

    # current camera placement for renderView1
    renderView1.CameraPosition = [5.754480303226986, -2.973217941582438, 1.6824212321685303]
    renderView1.CameraViewUp = [-0.22826967748316762, 0.10585123215171684, 0.9678266740453552]
    renderView1.CameraParallelScale = 1.7320508075688772

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(1090, 995)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.CameraPosition = [5.754480303226986, -2.973217941582438, 1.6824212321685303]
    renderView1.CameraViewUp = [-0.22826967748316762, 0.10585123215171684, 0.9678266740453552]
    renderView1.CameraParallelScale = 1.7320508075688772

    # save screenshot
    base_name, _ = os.path.splitext(xdmf_filename)
    png_filename = base_name + ".png"
    SaveScreenshot(filename=png_filename, viewOrLayout=renderView1, location=16, ImageResolution=[1570, 1054],
                   TransparentBackground=1,
                   SaveInBackground=1,
                   EmbedParaViewState=1)

###############################################################################
if __name__ == "__main__":
    '''Dump 3d data as a surface plot into a PNG image.

    Example of use:
       pvbatch paraview_surface_plot_screenshot_3d.py --filename data.xmf --field rho
    '''


    parser = argparse.ArgumentParser(description='Save surface plot into PNG image file.')
    parser.add_argument('--filename', type=str, default='data.xmf', help='kalypsso output data file in XDFM+HDF5 format')
    parser.add_argument('--field', type=str, default='rho', help='field name to display (e.g. rho)')
    args = parser.parse_args()

    dump_screenshot(args.filename, args.field)
