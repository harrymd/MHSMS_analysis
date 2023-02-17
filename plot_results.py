'''
plot_results.py
For plotting the data previously processed by process_data.py
Run from the command line:
    python3 plot_results.py
Run in the same directory as process_data.py.
'''

# Import necessary modules.
#
# Modules from Python standard library.
# datetime  Handles dates and times.
# os        For handling file paths in a platform-independent way.
# sys       For handling system-specific parameters and functions.
import datetime 
import json
import os
import sys
#
# Third-party modules.
# matplotlib The de-facto standard library for plotting.
# numpy     The de-facto standard library for numerical processing in Python.
# xarray    A library which extends NumPy arrays to allow labelled indices.
#           Note: See also the xarray optional dependencies
#               https://docs.xarray.dev/en/stable/getting-started-guide/installing.html#optional-dependencies
#           in particular we need either netCDF4 or SciPy for saving NetCDF
#           files (netCDF4 is preferable because SciPy doesn't support all
#           metadata attributes). These don't have to be imported but must
#           be installed.
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from scipy.stats import spearmanr
import xarray as xr

# Import optional modules.
# Third-party modules.
# mplcursors    Provides convenient wrappers for interactive plot cursors.
try:

    import mplcursors

except ModuleNotFoundError:

    print('Did not find module "mplcursors". Interactive point labels will not be enabled.')

# --- Generic -----------------------------------------------------------------
def load_nc_file(path_output_nc):

    # Load the NetCDF file.
    data = xr.open_dataarray(path_output_nc)

    # Restore the detailed names of the indices, which were converted from
    # dictionaries to strings during file saving.
    for dim in data.dims:

        try:
            
            data[dim].attrs['description'] = eval(data[dim].attrs['description'])

        except KeyError:

            pass

    return data

# --- Printing summaries. -----------------------------------------------------
def print_restraint_summaries(data, interventions = None):
    
    # Loop over the dimensions (axes) of the DataArray.
    print('Dimensions of dataset:')
    for dim in data.dims:
        
        print('{:>20} {:>6d}'.format(dim, len(data[dim])))
    
    # Find total number of observations, which is the product of the 
    # dimensions (e.g. if the array is 2 x 3 x 5, then there are 30
    # observations).
    n_obs = np.prod(data.shape)
    print('Total number of observations: {:>6d}'.format(n_obs))
    
    if interventions is not None:

        # Print more detail about a couple of interventions we are interested in.
        for intervention in interventions:
            
            print(80 * '-')
            print_summary_one_intervention_wrapper(data, intervention)

    print(80 * '-')
    # Print more detail about the data, combining all interventions.
    # To do this, we sum along the 'intervention' axis.
    data_all_interventions = data.sum(dim = 'intervention')
    print('\nSummary for all intervention types:')
    print_summary_one_dataset(data_all_interventions)

    return

def print_summary_one_intervention_wrapper(restraint_data, intervention):

    # Print the name of the intervention, and the description.
    intervention_description = \
        restraint_data.intervention.attrs['description'][intervention]
    print('\nSummary for intervention {:} ({:})'.format(intervention, intervention_description))

    # Select the data for this intervention, and print a summary of it.
    data = restraint_data.sel(intervention = intervention)
    print_summary_one_dataset(data)

    return

def print_summary_one_dataset(data):

    # Find the maximum value of the dataset, and the indices at which the
    # maximum value occurs. The method argmax() returns a dictionary where
    # the keys are the dimensions and the values are the indices. For
    # example, if the highest value occurred in January 2022 with provider
    # RX4, then
    #   argmax_dict['provider'] = 'RX4'
    #   argmax_dict['year_month'] = '2022_01'
    argmax_dict = data.argmax(...)
    # The value can be found using the dictionary of indices.
    # The isel() method returns another DataArray, but in this case it has
    # only one entry, so we discard the dimensions with squeeze(), and
    # convert to a Python scalar with item().
    argmax_value = data.isel(argmax_dict).squeeze().item()
    print('Maximum value: {:>6d} found at the following entry:'.format(argmax_value))
    
    # For each dimension, print the index at which the maximum value occurs,
    # also printing the long-form name of this index if they are given.
    for dimension, index_as_DataArray in argmax_dict.items():

        # As above, we use .squeeze().item() to reduce a single-valued DataArray
        # to a Python scalar.
        index = index_as_DataArray.squeeze().item()

        # Use the numerical index to extract the string coordinate.
        # For example if the coordinates along the 'year_month' axis are
        #   ['2021_09', '2021_10', ... ]
        # and the index is 1, then the coordinate is '2021_10'.
        coordinate_value = data.coords[dimension][index].squeeze().item()

        # Format the dimension and coordinate value.
        str_ = '{:<10} {:<10}'.format(dimension, coordinate_value)

        # Try to add the long form of the coordinate to the string.
        try:

            description = data[dimension].attrs['description'][coordinate_value]
            str_ = str_ + ' ' + description

        # If the long form is not found, just move on.
        except KeyError:
            
            pass

        # Print the string for this dimension.
        print(str_)
    
    ## The underlying NumPy array in the xarray DataArray can be accessed
    ## as the .values attribute, and then regular NumPy functions can be
    ## used such as np.mean().
    #mean_value = np.mean(data.values)
    #mean_value = data.mean().item()
    #print('Mean value: {:>8.2f}'.format(mean_value))

    # Find statistics about counts.
    total_count = data.sum().item() 
    n_providers = len(data['provider'])
    average_count_per_provider = total_count / n_providers
    # Get the sum of counts over all reporting periods.
    count_sum_over_year_month = data.sum('year_month')
    min_count_amongst_providers = count_sum_over_year_month.min().item()
    max_count_amongst_providers = count_sum_over_year_month.max().item()
    range_count_amongst_providers = max_count_amongst_providers - min_count_amongst_providers
    std_count_amongst_providers = count_sum_over_year_month.std().item()
    
    print('\n')
    print('Total count: {:8,d}'.format(total_count))
    print('Number of providers: {:>4d}'.format(n_providers))
    print('Average count: {:,.1f}'.format(average_count_per_provider))
    print('Minimum count: {:,d}'.format(min_count_amongst_providers))
    print('Maximum count: {:,d}'.format(max_count_amongst_providers))
    print('Range in count: {:,d}'.format(range_count_amongst_providers))
    print('Standard deviation in count: {:,.1f}'.format(std_count_amongst_providers))
    
    # Find the number of zero values (checking equality is not recommended
    # for floating-point numbers but here we are working with integers).
    # Also find the total number of values in the dataset, and report the
    # percentage which are zero.
    n_zeros = np.sum(data.values == 0)
    n_values = np.prod(data.shape)
    frac = n_zeros / n_values
    print('Number of zero values: {:>6d} out of {:>6d} ({:>5.2f} %)'.format(
        n_zeros, n_values, 100.0 * frac))

    return

def print_bed_days_summaries(data):
    
    # Loop over the dimensions (axes) of the DataArray.
    print('Dimensions of dataset:')
    for dim in data.dims:
        
        print('{:>20} {:>6d}'.format(dim, len(data[dim])))
    
    # Find total number of observations, which is the product of the 
    # dimensions (e.g. if the array is 2 x 3 x 5, then there are 30
    # observations).
    n_obs = np.prod(data.shape)
    print('Total number of observations: {:>6d}'.format(n_obs))
    
    ## Find statistics about number of bed days.
    #n_bed_days_tot = data.sum().item() 
    #n_providers = len(data['provider'])
    #average_bed_days_per_provider = n_bed_days_tot / n_providers
    ## Get the sum of bed-days overall all reporting periods.
    #bed_days_sum = data.sum('year_month')
    #min_bed_days_of_providers = bed_days_sum.min().item()
    #max_bed_days_of_providers = bed_days_sum.max().item()
    #range_bed_days_of_providers = max_bed_days_of_providers - min_bed_days_of_providers

    #print('Total number of bed days: {:8,d}'.format(n_bed_days_tot))
    #print('Number of providers: {:>4d}'.format(n_providers))
    #print('Average number of bed days: {:,.1f}'.format(average_bed_days_per_provider))
    #print('Minimum number of bed days: {:,d}'.format(min_bed_days_of_providers))
    #print('Maximum number of bed days: {:,d}'.format(max_bed_days_of_providers))
    #print('Range in number of bed days: {:,d}'.format(range_bed_days_of_providers))
    
    print_summary_one_dataset(data)

    return

# --- Generic plotting. -------------------------------------------------------
def generate_point_labels(fmt_string, list_of_flattened_vars):
    '''
    Loop over the input points and generate a label for each one, based on
    the input format.
    '''

    n_pts = len(list_of_flattened_vars[0])
    pt_label_list = []
    for i in range(n_pts):
        
        # Get all the variables for this input point.
        var_list_i = [x[i] for x in list_of_flattened_vars]

        # Format the variables into a label and add it to the list of labels.
        pt_label_i = fmt_string.format(*var_list_i)
        pt_label_list.append(pt_label_i)

    return pt_label_list

def get_point_colours(bed_days_data):
    '''
    Assign colours to points based on their hospital provider.
    '''
    
    # This dictionary defines the colours for each provider.
    # There are lots of providers with a small number of points, and these are
    # grouped together as 'other'.
    colour_dict = { 'RHA' : 'red',
                    'RX4' : 'green',
                    'RX3' : 'blue',
                    #'RXT' : 'purple',
                    'RW1' : 'purple',
                    'other' : 'black'}

    # The colours are stored in an XArray with the same shape as the 
    # bed-days data, and data type "object" (which allows strings to be stored).
    colours = xr.zeros_like(bed_days_data).astype(object)

    # We start by setting all the colours to the colour of "other" providers.
    # This will then be overriden for the providers specified in the dictionary.
    colours.loc[:] = colour_dict['other']

    # Loop over the providers specified as the keys of the dictionaries.
    for provider in colour_dict.keys():

        # We do not need to set the colour for 'other' providers because
        # that has already been done.
        if provider != 'other':

            # Set the colour for all entries matching the provider, using
            # the XArray .loc[] syntax.
            colours.loc[{'provider' : provider}] = colour_dict[provider]

    return colours, colour_dict

def scatter_plot(x, y, axis_labels, axis_lims, pt_labels = None, sizes = None,
        lines = None, fig_path = None, axes_types = ['lin', 'lin'],
        aspect = None, colours = None, colour_dict = None):
    '''
    A flexible wrapper script for plotting a scatter graph of x versus y.
    '''
    
    # Create the figure object.
    # 'constrained_layout' automatically adjusts the layout for consistent
    # spacing around the edges.
    fig_size_inches = (8.0, 6.0)
    fig = plt.figure(figsize = fig_size_inches, constrained_layout = True)
    ax  = plt.gca()
    
    # Define the properties of the scatter points.
    # These are stored in the `kwargs` (keyword arguments) dictionary with
    # the variables expected by PyPlot functions.
    alpha = 0.2
    kwargs = {'alpha' : alpha}
    # Set a default colour.
    if colours is None:

        colour = 'dodgerblue'
        kwargs['c'] = colour

    # Or use the colours provided by the user.
    else:

        kwargs['c'] = colours

    # Plot the scatter points.
    handle_scatter = ax.scatter(x, y, s = sizes,
                        clip_on = False, **kwargs)

    # If the colour dictionary was provided, create a legend describing the
    # meaning of the colours.
    if colour_dict is not None:
        
        for provider, colour in colour_dict.items():

            ax.scatter([], [], c = colour, label = provider, alpha = alpha)

        ax.legend(title = 'Provider')

    # Draw any additional lines specified.
    if lines is not None:
    
        # Set the line properties.
        line_kwargs = {'color' : 'k', 'alpha' : 0.5}

        for line in lines:

            x_line, y_line = line
            ax.plot(x_line, y_line, **line_kwargs)

    # Label the axes.
    font_size_label = 14
    x_label, y_label = axis_labels
    ax.set_xlabel(x_label, fontsize = font_size_label)
    ax.set_ylabel(y_label, fontsize = font_size_label)

    # Set the axes limits.
    x_lims, y_lims = axis_lims

    # If using linear x-axes, set the axis limits. Otherwise set the axes
    # to logarithmic and use automatic axis limits.
    if axes_types[0] == 'lin':

        ax.set_xlim(x_lims)

    elif axes_types[0] == 'log':
        
        ax.set_xscale('log')

    else:

        raise ValueError

    # Same as above, for y-axis.
    if axes_types[1] == 'lin':

        ax.set_ylim(y_lims)

    elif axes_types[1] == 'log':

        ax.set_yscale('log')

    else:
        
        raise ValueError

    # Set the aspect ratio, if requested.
    if aspect is not None:

        ax.set_aspect(aspect)

    # Save the figure (if a path is provided).
    if fig_path is not None:

        print("Saving to {:}".format(fig_path))
        plt.savefig(fig_path, dpi = 300)

    # Add interactive point labels (if mplcursors has been imported).
    if pt_labels is not None:

        if 'mplcursors' in sys.modules:

            # Create the cursor object which responds to hovering.
            cursor = mplcursors.cursor(handle_scatter, hover = True)

            # Define the response of the cursor to hovering over a point.
            # (The response is to show the pre-defined label for that point.)
            @cursor.connect("add")
            def on_add(sel):

                # Get the label for this point.
                pt_label = pt_labels[sel.index]

                # Set the label for this point.
                sel.annotation.set(text = pt_label)

    return

def make_label_list_common(array):

    # Create a dictionary which stores all of the variables which go into each
    # label.
    pt_label_dict = dict()

    # Generate a grid of reporting period and provider name corresponding to
    # the coordinates of the XArray (note: XArray probably has a built-in
    # function to do this, but I haven't been able to find it).
    pt_label_dict['year_month'], pt_label_dict['provider'] = \
            np.meshgrid(list(array.year_month.values),
                        list(array.provider.values), indexing = 'ij')

    # Generate a grid of provider names and provider descriptions corresponding
    # to the coordinates of the XArray.
    prov_desc_for_label = np.zeros(pt_label_dict['provider'].shape, dtype = object)
    for i in range(prov_desc_for_label.shape[0]):

        for j in range(prov_desc_for_label.shape[1]):

            prov = pt_label_dict['provider'][i, j]
            val = array.provider.attrs['description'][prov]
            prov_desc_for_label[i, j] = val

    return pt_label_dict, prov_desc_for_label

# --- Two-intervention scatter plots. -----------------------------------------
def make_label_list_for_two_intervention_plot(restraint_data, x, y,
                        norm_tot_int, x_norm_tot_int, y_norm_tot_int,
                        norm_bed_days, x_norm_bed_days, y_norm_bed_days):
    '''
    We generate labels for each point and will make them appear when hovering
    over that point with the mouse, which can make it easier to inspect the
    data.
    The labels contain the reporting period, provider, data counts and
    percentages.
    The labels must be provided as a "flattened" list. For example if we
    have a dataset with N reporting periods and M providers, the labels
    will be a list of length (M * N). The flattening must be done carefully
    to avoid mismatching the labels and the points.

    This function is very similar to make_label_llist_for_bed_day_versus_intervention_plot().
    '''

    # Generate lists of reporting period, provider and provider description
    # on a grid corresponding to the data.
    pt_label_dict, prov_desc_for_label = make_label_list_common(restraint_data)

    # Store all of the arrays of  variables which are required to generate the 
    # labels, and flatten them.
    list_of_flattened_vars_for_pt_labels = \
        [   pt_label_dict['year_month'],
            pt_label_dict['provider'], prov_desc_for_label,
            x, 100.0 * x_norm_tot_int, 1000.0 * x_norm_bed_days,
            y, 100.0 * y_norm_tot_int, 1000.0 * y_norm_bed_days,
            norm_tot_int, norm_bed_days]
    #
    list_of_flattened_vars_for_pt_labels = [x.flatten() for x in
                                            list_of_flattened_vars_for_pt_labels]

    # Define the format of the labels.
    fmt_str_for_pt_labels = 'Reporting period: {:}'\
                            '\nProvider: {:} ({:})'\
                            '\nOral chemical count: {:>3d} ({:>6.2f} % of interventions, {:>6.2f} per 1000 bed days)'\
                            '\nSeclusion count: {:>3d} ({:>6.2f} % of interventions, {:>6.2f} per 1000 bed days)'\
                            '\nTotal restraint count: {:>3d}'\
                            '\nBed days: {:>3d}'

    # Generate the labels.
    pt_label_list = generate_point_labels(fmt_str_for_pt_labels,
                        list_of_flattened_vars_for_pt_labels)

    return pt_label_list

def scatter_plot_two_intervention_types(dir_plot, restraint_data, bed_days_data, intervention_code_1, intervention_code_2, plot_var, axes_types, colours, colour_dict):

    # Make a scatter plot of
    #   x: Number of patients subject to oral chemical restraint (measure 14);
    #   y: Number of patients subject to seclusion (measure 5);
    # where each scatter point for both x and y is for a given month and
    # a given provider, and (optionally) each point is normalised by the
    # total number of patients restrained (by any means) for the given month
    # and given provider.
    data_all_interventions = restraint_data.sum(dim = 'intervention')
    intervention_1_data = restraint_data.sel(intervention = intervention_code_1)
    intervention_2_data = restraint_data.sel(intervention = intervention_code_2)
    #
    x = intervention_1_data.values
    y = intervention_2_data.values
    
    # Calculate norms.
    norm_tot_int = data_all_interventions.values
    norm_bed_days = bed_days_data.values

    # Normalise values.
    x_norm_tot_int = x / norm_tot_int
    y_norm_tot_int = y / norm_tot_int
    #
    x_norm_bed_days = x / norm_bed_days
    y_norm_bed_days = y / norm_bed_days

    ## Note that dividing by the total number of interventions will produce
    ## a NumPy RunTimeWarning for division by zero, because some entries are
    ## zero. The resulting values with be np.nan (not a number), which are then
    ## ignored by the plotting commands.

    # Generate labels for each point.
    pt_label_list = make_label_list_for_two_intervention_plot(restraint_data,
                        x, y,
                        norm_tot_int, x_norm_tot_int, y_norm_tot_int,
                        norm_bed_days, x_norm_bed_days, y_norm_bed_days)
    
    # In both types of plots, the axes start at 0 (counts cannot be negative).
    x_lim_min = 0.0
    y_lim_min = 0.0
    
    # Set properties for count-type plots.
    if plot_var == 'count':

        x_var = x
        y_var = y
        x_label = 'Number of interventions using oral chemical restraint'
        y_label = 'Number of interventions using seclusion'
        fig_name = 'scatter_oral_chem_seclusion_count_{:}{:}.png'.format(*axes_types)

        # Here the axis limits are simply high enough to fit the largest values.
        x_lim_max = np.nanmax(x) * 1.1
        y_lim_max = np.nanmax(y) * 1.1
        aspect = None

        pt_sizes = None

        lines = None

    # Set properties for ratio-type plots.
    elif plot_var == 'norm_tot_int':

        x_var = x_norm_tot_int
        y_var = y_norm_tot_int
        x_label = 'Fraction of interventions using oral chemical restraint'
        y_label = 'Fraction of interventions using seclusion'
        fig_name = 'scatter_ratio_oral_chem_seclusion_norm_tot_int_{:}{:}.png'.format(*axes_types)

        # The axis limits are large enough to fit the highest values, but
        # do not go above 1.
        x_lim_max = np.min([np.nanmax(x_norm_tot_int) * 1.1, 1.0])
        y_lim_max = np.min([np.nanmax(y_norm_tot_int) * 1.1, 1.0])
        aspect = None

        # For ratio plots, the sum of the two variables cannot exceed 1.
        # This creates a 'forbidden' region of the plot, which we mark
        # with a line.
        lines = [[[0.0, 1.0], [1.0, 0.0]]]

        pt_sizes = norm_tot_int * 0.5

    # Set properties for ratio-type plots.
    elif plot_var == 'norm_bed_days':

        x_var = 1000.0 * x_norm_bed_days
        y_var = 1000.0 * y_norm_bed_days
        x_label = 'Number of interventions using oral chemical restraint per 1000 bed days'
        y_label = 'Number of interventions using seclusion per 1000 bed days'
        fig_name = 'scatter_oral_chem_seclusion_norm_bed_days_{:}{:}.png'.format(*axes_types)

        x_lim_max = np.nanmax(x_var) * 1.1
        y_lim_max = np.nanmax(y_var) * 1.1

        # Manually override limits to zoom in on bulk of data.
        x_lim_max = 6.0
        y_lim_max = 6.0
        xy_lim_max = np.max([x_lim_max, y_lim_max])

        # Set aspect ratio to 1.
        aspect = 'equal'

        # Define size of points.
        pt_sizes = norm_bed_days * 0.005

        # Define 1:1 line.
        lines = [[[0.0, xy_lim_max], [0.0, xy_lim_max]]]

    else:

        raise ValueError
    
    # Store the axes limits and labels.
    axis_lims = [[x_lim_min, x_lim_max], [y_lim_min, y_lim_max]]
    axis_labels = [x_label, y_label]

    no_nan = (~np.isnan(x_var) & ~np.isnan(y_var))
    spearman_res = spearmanr(x_var[no_nan], y_var[no_nan])
    #spearman_res = spearmanr([1, 2, 3, 4, 5], [5, 6, 7, 8, 7], nan_policy = 'omit')
    print("Spearman's rank correlation coefficient and p-value:")
    print(spearman_res)

    # Make the scatter plot.
    fig_path = os.path.join(dir_plot, fig_name)
    scatter_plot(x_var, y_var, axis_labels, axis_lims,
            pt_labels = pt_label_list, sizes = pt_sizes,
            lines = lines, fig_path = fig_path, axes_types = axes_types,
            aspect = aspect, colours = colours.values.flatten(), colour_dict = colour_dict)

    return

# --- Bed-day versus intervenion scatter plots. -------------------------------
def make_label_list_for_bed_day_versus_intervention_plot(bed_days_filtered, intervention_data, x, y):
    '''
    We generate labels for each point and will make them appear when hovering
    over that point with the mouse, which can make it easier to inspect the
    data.
    The labels contain the reporting period, provider, data counts and
    percentages.
    The labels must be provided as a "flattened" list. For example if we
    have a dataset with N reporting periods and M providers, the labels
    will be a list of length (M * N). The flattening must be done carefully
    to avoid mismatching the labels and the points.

    This function is very similar to make_label_list_for_two_intervention_plot().
    '''

    # Generate lists of reporting period, provider and provider description
    # on a grid corresponding to the data.
    pt_label_dict, prov_desc_for_label = make_label_list_common(bed_days_filtered)

    # Store all of the arrays of  variables which are required to generate the 
    # labels, and flatten them.
    list_of_flattened_vars_for_pt_labels = \
        [   pt_label_dict['year_month'], pt_label_dict['provider'],
            prov_desc_for_label, np.round(1000.0 * x).astype(int), y]
    #
    list_of_flattened_vars_for_pt_labels = [x.flatten() for x in
                                            list_of_flattened_vars_for_pt_labels]

    # Define the format of the labels.
    fmt_str_for_pt_labels = 'Reporting period: {:}'\
                            '\nProvider: {:} ({:})'\
                            '\nBed-day count: {:>3d}'\
                            '\nOral chemical count: {:>3d}'

    # Generate the labels.
    pt_label_list = generate_point_labels(fmt_str_for_pt_labels,
                        list_of_flattened_vars_for_pt_labels)

    return pt_label_list

def scatter_plot_bed_days_versus_intervention_type(dir_plot, bed_days_data, restraint_data, intervention, axes_types):

    # Get the data relating to the intervention type of interest (e.g. oral
    # chemical restraint).
    intervention_data = restraint_data.sel(intervention = intervention).squeeze()

    # Convert to 1000 bed days.
    x = bed_days_data.values * 1.0E-3

    # Filler values (negative) can be replaced with NaN.
    x[x < 0.0] = np.nan

    # Get the values from the array.
    y = intervention_data.values

    # Generate the labels for the points.
    pt_labels = make_label_list_for_bed_day_versus_intervention_plot(
            bed_days_data, intervention_data, x, y)

    # Set the axes limits.
    x_lim_min = 0.0
    y_lim_min = 0.0
    x_lim_max = np.nanmax(x) * 1.1
    y_lim_max = np.nanmax(y) * 1.1
    axis_lims = [[x_lim_min, x_lim_max], [y_lim_min, y_lim_max]]

    # Set the axes labels.
    x_label = 'Bed days (thousands)'
    y_label = 'Number of interventions using oral chemical restraint'
    axis_labels = [x_label, y_label]
    
    # Choose linear axes (not logarithmic).
    axes_type = 'linear'

    # Define the name of the output file.
    fig_name = 'scatter_bed_days_oral_chem_{:}{:}.png'.format(*axes_types)
    fig_path = os.path.join(dir_plot, fig_name)

    # Generate the scatter plot.
    scatter_plot(x, y, axis_labels, axis_lims,
            pt_labels = pt_labels,
            sizes = None,
            lines = None,
            fig_path = fig_path,
            axes_types = axes_types)

    return

# --- Bed-day grid plot. ------------------------------------------------------
def summarise_incomplete_reporting(restraint_data, intervention_not_reported,
        bed_days_sum_sorted):

    # Get counts of number of providers not reporting or partially reporting,
    # and how many months are not reported.
    providers_not_reporting = intervention_not_reported.all(dim = 'year_month')
    providers_partially_reporting = intervention_not_reported.any(dim = 'year_month')
    n_months_missing = intervention_not_reported.sum(dim = 'year_month')
    
    # Print a summary on missing reporting.
    fmt = '{:>20} {:<15} {:<60}'
    len_fmt = len(fmt.format('', '', '', ''))
    line_str = '-' * len_fmt
    print("\nThe following providers did not report oral chemical intervention for any reporting period:")
    # First, print header line, underlined.
    print(fmt.format('Number of bed days', 'Provider code', 'Provider full name'))
    print(line_str)
    # Then print by provider. 
    for prov_array in providers_not_reporting:
        
        # Only print if the provider is not reporting.
        prov_is_not_reporting = prov_array.item()
        if prov_is_not_reporting:
            
            prov = prov_array.provider.item()
            n_bed_days =  bed_days_sum_sorted.loc[prov].item()
            desc = restraint_data.provider.attrs['description'][prov]
            print(fmt.format(n_bed_days, prov, desc))

    # Print a similar summary on partial reporting.
    fmt = '{:>20} {:<15} {:<60} {:>15}'
    len_fmt = len(fmt.format('', '', '', ''))
    line_str = '-' * len_fmt
    print("\nThe following providers did not report oral chemical intervention for some reporting periods:")
    # First, print header line, underlined.
    print(fmt.format('Number of bed days', 'Provider code', 'Provider full name', 'Months missing'))
    print(line_str)
    # Then print by provider. 
    for prov_not_array, prov_partial_array in zip(providers_not_reporting, providers_partially_reporting):
        
        prov_is_not_reporting_any = prov_not_array.item()
        prov_is_not_reporting_some = prov_partial_array.item()
        
        # Only print if the provider is partially reporting.
        prov = prov_partial_array.provider.item()
        if (not prov_is_not_reporting_any) and (prov_is_not_reporting_some):
            
            prov = prov_partial_array.provider.item()
            n_bed_days =  bed_days_sum_sorted.loc[prov].item()
            desc = restraint_data.provider.attrs['description'][prov]
            n_months = n_months_missing.loc[prov].item()
            print(fmt.format(n_bed_days, prov, desc, n_months))

    return

def grid_plot_bed_days(dir_plot, bed_days_data, restraint_data, intervention):
    '''
    Makes a plot which summarises the reporting of a particular intervention,
    highlighting providers that are under-reporting.
    '''

    # Get the data relating to the intervention type of interest (e.g. oral
    # chemical restraint).
    intervention_data = restraint_data.sel(intervention = intervention).squeeze()

    # Get the sum of bed-days overall all reporting periods.
    bed_days_sum = bed_days_data.sum('year_month')

    # Sort data by total bed-days.
    i_sort = (-1 * bed_days_sum).argsort()
    bed_days_sum_sorted = bed_days_sum.isel({'provider' : i_sort.values})
    bed_days_sorted = bed_days_data.isel({'provider' : i_sort.values})
    intervention_data_sorted = intervention_data.isel({'provider' : i_sort.values})

    # Identify reporting periods where there is no reporting of a given 
    # intervention.
    intervention_not_reported = (intervention_data_sorted == 0) | (intervention_data_sorted == -1)

    # Print summary about incomplete reporting.
    summarise_incomplete_reporting(restraint_data, intervention_not_reported,
            bed_days_sum_sorted)

    # Create the figure object.
    # 'constrained_layout' automatically adjusts the layout for consistent
    # spacing around the edges.
    fig_size_inches = (8.0, 10.0)
    fig = plt.figure(figsize = fig_size_inches, constrained_layout = True)
    ax  = plt.gca()

    # We combine PyPlot's imshow() with NumPy's masked arrays to make a
    # bitmap plot where the missing values are highlighted in a different
    # colour.
    # The array is masked at every reporting period with no intervention
    # reported.
    X = np.ma.masked_where(intervention_not_reported, intervention_data_sorted)
    # Use a grey palette, with missing reporting highlighted in red.
    palette = plt.cm.gray.with_extremes(bad = 'r')
    # Set the range of the palette to span the range of the data. 
    c_norm = Normalize(vmin = 0.0, vmax = np.ma.max(X))
    
    # Create the bitmap image.
    handle_im = ax.imshow(X.T, cmap = palette, norm = c_norm)

    # Add a small set of axes to the right of the main axes to house the
    # colour bar.
    ax_divider = make_axes_locatable(ax)
    c_ax = ax_divider.append_axes("bottom", size = "3%", pad = "1%")

    # Create the colour bar in the new axes.
    cb = fig.colorbar(handle_im, cax = c_ax, orientation = 'horizontal',
            label = 'Oral chemical restraint')
    
    # Define custom 'tick labels' down the y-axis which contain the provider
    # code, total bed days for that provider, and full name of provider.
    label_style = 'anonymous'
    #label_style = 'full'
    n_providers = len(bed_days_sum)
    y_ticks = list(range(n_providers))
    y_tick_labels = []
    for i in range(n_providers):
        
        if label_style == 'full':

            provider = bed_days_sorted.coords['provider'][i].squeeze().item()
            provider_description = \
                bed_days_sorted['provider'].attrs['description'][provider]
            
            tick_label_i = '{:6} ({:>7,}) {:}'.format(provider,
                                bed_days_sum_sorted[i].item(), provider_description)

        elif label_style == 'anonymous':

            tick_label_i = '{:>7,}'.format(bed_days_sum_sorted[i].item())

        else:

            raise ValueError

        y_tick_labels.append(tick_label_i)
    #
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    # Label axes.
    ax.set_xlabel('Reporting period', fontsize = 14)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel('Provider size (bed days)', fontsize = 14)
    ax.yaxis.set_label_position("right")
    
    # Label the reporting periods along the x-axis.
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                    'Sep', 'Oct', 'Nov', 'Dec']
    int_str_to_month = {"{:>02d}".format(i + 1) : month for 
                        i, month in enumerate(month_list)}
    year_int_str_prev = '1900'
    x_tick_labels = []
    for year_month_day in bed_days_sorted.coords['year_month']:
        
        label_raw_string = year_month_day.item()
        year_int_str, month_int_str = label_raw_string.split('_')
        month_str = int_str_to_month[month_int_str]
        if (year_int_str != year_int_str_prev):
            label = '{:} {:}'.format(month_str, year_int_str)
        else:
            label = month_str

        #x_tick_labels.append(year_month_day.item())
        x_tick_labels.append(label)
        year_int_str_prev =  year_int_str

    #
    x_ticks = list(range(len(bed_days_sorted.coords['year_month'])))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation = 90)

    # Move x-labels to top.
    # Move y-labels to right.
    # Set label font to monospace.
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    #
    ticks_font = font_manager.FontProperties(family = 'monospace')
    labels_xy = [ax.get_xticklabels(), ax.get_yticklabels()]
    for labels in labels_xy:

        for label in labels:

            label.set_fontproperties(ticks_font)

    # Define output file.
    fig_name = 'grid_oral_chem.png'
    fig_path = os.path.join(dir_plot, fig_name)

    # Save figure.
    print("Saving to {:}".format(fig_path))
    plt.savefig(fig_path, dpi = 300, bbox_inches = 'tight')

    return

# --- Main. -------------------------------------------------------------------
def main():

    # Define some file and directory paths.
    # We use the 'os' module to create the file path string from its components.
    dir_output = 'output'
    dir_plot = os.path.join(dir_output, 'plots')
    file_restraint_nc = 'restraint_data.nc'
    path_restraint_nc = os.path.join(dir_output, file_restraint_nc)
    file_bed_days_nc = 'bed_days_data.nc'
    path_bed_days_nc = os.path.join(dir_output, file_bed_days_nc)

    # Check the plot directory exists.
    err_str = 'The plot directory ({:}) does not exist. Please create it.'.format(dir_plot)
    assert os.path.isdir(dir_plot), err_str

    # Load the xarray DataArrays.
    restraint_data = load_nc_file(path_restraint_nc)
    bed_days_data  = load_nc_file(path_bed_days_nc)

    # Combine the restraint data and bed-days data.
    # (Note: data from non-overlapping providers/months will be discarded.)
    bed_days_data, restraint_data = xr.align(bed_days_data, restraint_data)

    # Get colours of scatter points.
    colours, colour_dict = get_point_colours(bed_days_data)

    # Print some summaries of the data.
    #print(restraint_data)
    #for intervention in restraint_data.coords['intervention']:
    #    
    #    int_code = intervention.item()
    #    print(int_code, restraint_data.coords['intervention'].attrs['description'][int_code])

    print(80 * '-')
    print('Summary of restraint data:')
    print_restraint_summaries(restraint_data, interventions = ['05', '14', '15', '16', '17'])

    print(80 * '-')
    print('\n\nSummary of bed-days data:')
    print_bed_days_summaries(bed_days_data)

    # Set which types of plots you wish to generate..
    plot_types = ['two_interventions', 'bed_days_vs_intervention', 'bed_days_grid']
    #plot_types = ['two_interventions', 'bed_days_vs_intervention']
    #plot_types = ['bed_days_vs_intervention']
    #plot_types = ['two_interventions']
    plot_types = ['bed_days_grid']
    for plot_type in plot_types:

        if plot_type == 'two_interventions':

            # Make a scatter plot comparing two intervention types across reporting
            # period and provider.
            # For 'plot_var', choose between:
            # 'count'           Raw counts.
            # 'norm_tot_int'    Counts divided by number of interventions of
            #                   any kind.
            # 'norm_bed_days'    Counts divided by number of bed days.
            intervention_1 = '14' # '14' is oral chemical intervention.
            intervention_2 = '05' # '05' is seclusion intervention.
            plot_var = 'norm_bed_days'
            #axes_types = ['log', 'log'] # 'lin' (linear) or 'log' for x and y axes.
            axes_types = ['lin', 'lin'] # 'lin' (linear) or 'log' for x and y axes.
            scatter_plot_two_intervention_types(dir_plot, restraint_data, bed_days_data,
                    intervention_1, intervention_2, plot_var, axes_types, colours,
                    colour_dict)

        elif plot_type == 'bed_days_vs_intervention':

            # Make a scatter plot of bed-days against number of interventions of
            # a specified type.
            intervention = '14' # '14' is oral chemical intervention.
            #axes_types = ['log', 'log'] # 'lin' (linear) or 'log' for x and y axes.
            axes_types = ['lin', 'lin'] # 'lin' (linear) or 'log' for x and y axes.
            scatter_plot_bed_days_versus_intervention_type(dir_plot, bed_days_data,
                    restraint_data, intervention, axes_types)

        elif plot_type == 'bed_days_grid':

            # Make a bitmap plot summarising the reporting of an intervention
            # across reporting period and provider.
            intervention = '14' # '14' is oral chemical intervention.
            grid_plot_bed_days(dir_plot, bed_days_data, restraint_data, intervention)

        else:

            raise ValueError('Plot type {:} not recognised'.format(plot_type))

    # Show any plots which have been created.
    plt.show()

    return

if __name__ == '__main__':

    main()
