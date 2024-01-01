'''
process_data.py
For extracting and processing subsets of data from NHS summary CSV files.
Run from the command line:
    python3 process_data.py
The input files are expected to be in directories one level below the directory
containing this script:
    ../Main database Oct 2021 to September 2022_tidied
    ../Restraint Data Oct 2021 to September 2022_tidied
'''

# Import necessary modules.
# 
# Modules from Python standard library.
# glob  Allows "globbing" of files (identifying multiple files with similar
#       file-name patterns).
# os    For handling file paths in a platform-independent way.
from glob import glob
import os
#
# Third-party modules.
# numpy     The de-facto standard library for numerical processing in Python.
# pandas    The de-facto standard library data analysis involving tables in Python.
# xarray    A library which extends NumPy arrays to allow labelled indices.
#           Note: See also the xarray optional dependencies
#               https://docs.xarray.dev/en/stable/getting-started-guide/installing.html#optional-dependencies
#           in particular we need either netCDF4 or SciPy for saving NetCDF
#           files (netCDF4 is preferable because SciPy doesn't support all
#           metadata attributes). These don't have to be imported but must
#           be installed.
import numpy as np
import pandas as pd
import xarray as xr

# Global variables.
#
# Use dict comprehensions to create dictionary mappings between month
# integer and month string and vice versa, e.g.
#   month_int_to_str[1]     = 'Jan'
#   month_str_to_int['Jan'] = 1
month_str_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                    'Sep', 'Oct', 'Nov', 'Dec']
month_int_to_str = {i + 1 : month_str for i, month_str in enumerate(month_str_list)}
month_str_to_int = {month_str : i + 1 for i, month_str in enumerate(month_str_list)}
#
# Define format for year-month strings, e.g. March 2022 --> '2022_03'.
year_month_fmt = '{:4d}_{:02d}'

# -----------------------------------------------------------------------------
def get_list_of_input_year_month_pairs(dir_data, file_str_break_points):
    
    # Use glob to find all the input files in the input directory.
    path_input_wildcard = os.path.join(dir_data, '*.csv')
    path_input_list = glob(path_input_wildcard)
    n_input_files = len(path_input_list)

    # Create a NumPy array listing all of the year-month pairs in the dataset.
    year_month_pairs = np.zeros((n_input_files, 2), dtype = int)
    for i in range(n_input_files):

        # Convert from path to file name, e.g.
        #   '/path/to/my_file.txt' -> 'my_file'
        name_file = os.path.splitext(os.path.basename(path_input_list[i]))[0]

        # Get the month and year as strings.
        # Warning: This depends on the exact file-naming convention being used.
        month_str = name_file[file_str_break_points[0] : file_str_break_points[0] + 3]
        year_str = name_file[file_str_break_points[1] : file_str_break_points[1] + 4]
        
        # Convert month and year strings to integers and store.
        month_int = month_str_to_int[month_str]
        year_int = int(year_str)
        year_month_pairs[i, 0] = year_int
        year_month_pairs[i, 1] = month_int

    # Sort arrays so that they are sorted first by year and then by month.
    year_month_pairs = year_month_pairs[np.lexsort((year_month_pairs[:, 1],
                                                    year_month_pairs[:, 0]))]

    # Print information about input files.
    print_year_month_pair_info(year_month_pairs)

    return year_month_pairs

def print_year_month_pair_info(year_month_pairs):
    '''
    Prints the sorted year-month pairs, also highlighting any gaps.
    '''

    n_input_files = len(year_month_pairs)
    print('Found {:} input files.'.format(n_input_files))

    if year_month_pairs[-1, 1] == 12:

        year_end = year_month_pairs[-1, 0] + 1
        month_end = 1

    else:

        year_end = year_month_pairs[-1, 0]
        month_end = year_month_pairs[-1, 1] + 1

    i = 0
    year = year_month_pairs[0, 0]
    month = year_month_pairs[0, 1]

    while not ((year == year_end) and (month == month_end)):

        if (year_month_pairs[i, 0] == year) and (year_month_pairs[i, 1] == month):

            print('{:4d}. {:4d} {:2d}: File found'.format(i + 1, year, month))
            i = i + 1

        else:

            print('      {:4d} {:2d}: File not found'.format(year, month))
        
        if month == 12:

            month = 1 
            year = year + 1

        else:

            month = month + 1

    print('')

    return

def load_csv_from_year_month(dir_data, file_name_format, year, month):

    # Load the input file for this year-month pair.
    month_str = month_int_to_str[month]
    file_name = file_name_format.format(month_str, year)
    file_path = os.path.join(dir_data, file_name)

    # The file is loaded into a Pandas data frame.
    # The input files appear to be saved in the 'cp1252' encoding (Windows).
    # If the 'low_memory' flag is not set to False, we may see a warning
    # about the last column having mixed data types (e.g. '*' and 31.25).
    # Pandas can handle this, so we just ignore it by switching the flag off.
    df_input = pd.read_csv(file_path, encoding = 'cp1252', low_memory = False)

    return df_input

def reduce_to_unique_sorted_list(list_, description_list):

    # Convert the list to a NumPy array so we can use the np.unique function.
    # Then use np.unique to sort the array and extract the unique values.
    # It also provides a list of indices so corresponding values from other
    # arrays can also be extracted.
    array = np.array(list_, dtype = object)
    array, i_unique = np.unique(array, return_index = True)

    # Use the list of indices to extract the descriptions corresponding to the
    # unique, sorted values obtained above.
    description_array = np.array(description_list, dtype = object)
    description_array = description_array[i_unique]

    return array, description_array

# --- Handling restraint CSV files. -------------------------------------------
def get_list_of_providers_and_interventions_from_restraint_csv(dir_data, year_month_pairs, file_name_format):

    # Loop over the year-month pairs to get lists of providers and
    # intervention types. We first collect all of the listed values, then
    # reduce these lists to just the unique values. We also store the
    # descriptions (e.g. intervention '14' has description
    # 'Chemical restraint - Oral').
    #
    # Prepare the empty lists.
    providers_list = []
    providers_description_list = []
    interventions_list = []
    interventions_description_list = []
    #
    # Loop over the number of monthly reporting periods.
    n_monthly_reports = year_month_pairs.shape[0]
    for i in range(n_monthly_reports):

        # Load the input file for this year-month pair and store it in a
        # Pandas data frame.
        df_input = load_csv_from_year_month(dir_data,
                                            file_name_format,
                                            *year_month_pairs[i, :])
        
        # At some point, the column labels were changed in the NHS MHS.
        # Use this mapping to convert the old labels to new style.
        col_rename_dict = { 'PRIMARY_LEVEL'                 : 'LEVEL_ONE',
                            'PRIMARY_LEVEL_DESCRIPTION'     : 'LEVEL_ONE_DESCRIPTION',
                            'SECONDARY_LEVEL'               : 'LEVEL_TWO',
                            'SECONDARY_LEVEL_DESCRIPTION'   : 'LEVEL_TWO_DESCRIPTION',
                            }
        df_input = df_input.rename(columns = col_rename_dict)

        # Get a list of all providers mentioned in this file, and append to the
        # master list.
        # In Pandas, if the dataframe has a column e.g. 'BREAKDOWN', it can
        # be accessed as follows: dataframe.BREAKDOWN
        key = 'Provider'
        df_providers = df_input[df_input.BREAKDOWN == key]

        providers_list.extend(list(df_providers.LEVEL_ONE))
        providers_description_list.extend(list(df_providers.LEVEL_ONE_DESCRIPTION))

        # Get a list of all intervention types mentioned in this file, and
        # append to the master list.
        key = 'Provider; Restrictive Intervention Type'
        df_intervention = df_input[df_input.BREAKDOWN == key]

        interventions_list.extend(list(df_intervention.LEVEL_TWO))
        interventions_description_list.extend(list(df_intervention.LEVEL_TWO_DESCRIPTION))

    # Reduce the lists of providers and interventions to unique, sorted lists.
    providers, providers_description = \
        reduce_to_unique_sorted_list(providers_list, providers_description_list)
    interventions, interventions_description = \
        reduce_to_unique_sorted_list(interventions_list, interventions_description_list)

    # The intervention codes are sometimes stored in the format '1', '2' ... and
    # sometimes in the format '01', '02', ... so these need to be merged.
    interventions_list = []
    for intervention in interventions:
        
        # The special code 'UNKNOWN' is left unchanged.
        if intervention != 'UNKNOWN':

            # Cast as int, then format as zero-padded string, e.g.
            #   '01' -> 1 -> '01'
            #   '1' -> 1 -> '01' 
            intervention = '{:>02d}'.format(int(intervention))
        
        interventions_list.append(intervention)
    
    # For a second time, remove duplicate values from interventions list.
    interventions_description_list = list(interventions_description)
    interventions, interventions_description = \
        reduce_to_unique_sorted_list(interventions_list, interventions_description_list)

    # Report number of providers and interventions identified.
    n_providers = len(providers)
    n_interventions = len(interventions)
    print('Found {:d} providers'.format(n_providers))
    print('Found {:d} interventions'.format(n_interventions))

    return providers, interventions, providers_description, interventions_description

def prepare_restraint_data_output_array(year_month_pairs, interventions, providers, interventions_description, providers_description):

    # Create a list of strings which combine the year-month pairs to act as an index.
    year_month_str_list = [year_month_fmt.format(*year_month) for year_month
                                in year_month_pairs]

    # Get the dimensions of the output array.
    n_providers = len(providers)
    n_monthly_reports = len(year_month_str_list)
    n_interventions = len(interventions)

    # Prepare the output, an xarray DataArray.
    # It has three dimensions: intervention type, provider, and month.
    # So, for example, if the intervention type is 5 (Seclusion), the month is
    # October 2021, and the provider is DE8 (Elysium Healthcare), then the
    # value (number of people) can be accessed with this syntax 
    #   n_people = restraint_data.sel(  intervention = '05',
    #                                   year_month = '2021_10',
    #                                   provider = 'DE8')
    # and it can be updated with this syntax:
    #   restraint_data.loc['05', '2021_10', 'DE8'] = n_people.
    # See the function print_summaries() in plot_results.py for more
    # examples of manipulating the DataArray.
    empty_array = np.zeros((n_interventions, n_monthly_reports, n_providers),
                        dtype = int)
    restraint_data = xr.DataArray(empty_array,
            coords = [list(interventions), year_month_str_list, list(providers)],
            dims = ['intervention', 'year_month', 'provider'])

    # Set the attributes of the data array (these are just labels, but can
    # help to make data more readable).
    # Note: These may not be saved properly if the netCDF Python library isn't
    # installed.
    restraint_data.attrs["long_name"] = 'Restraint data: number of patients'
    restraint_data.attrs["units"] = "patients"
    restraint_data.attrs["description"] = 'Metric MHS76 (number of people subject to '\
        'restrictive intervention), gridded by intervention type, provider '\
        'and monthly reporting period.'

    # We store the full variable names as strings, because they are useful later
    # when inspecting the data.
    # (They should be dictionaries but saving dictionary attributes of NetCDF
    # files is not supported; they can be converted back to dicts when loading
    # the files later.)
    restraint_data.intervention.attrs['description'] = str({int_ : int_desc
            for int_, int_desc in zip(interventions, interventions_description)})
    restraint_data.provider.attrs['description'] = str({int_ : int_desc 
            for int_, int_desc in zip(providers, providers_description)})

    return restraint_data

def parse_one_restraint_csv_file(dir_data, file_name_format, year, month, restraint_data):

    # Load the input file for this year-month pair and store it in a
    # Pandas data frame.
    df_input = load_csv_from_year_month(dir_data, file_name_format, year, month)

    # At some point, the column labels were changed in the NHS MHS.
    # Use this mapping to convert the old labels to new style.
    col_rename_dict = { 'PRIMARY_LEVEL'                 : 'LEVEL_ONE',
                        'PRIMARY_LEVEL_DESCRIPTION'     : 'LEVEL_ONE_DESCRIPTION',
                        'SECONDARY_LEVEL'               : 'LEVEL_TWO',
                        'SECONDARY_LEVEL_DESCRIPTION'   : 'LEVEL_TWO_DESCRIPTION',
                        'MEASURE_ID'                    : 'METRIC',
                        'MEASURE_VALUE'                 : 'METRIC_VALUE',
                        }
    df_input = df_input.rename(columns = col_rename_dict)

    # Get the year-month index.
    year_month_str = year_month_fmt.format(year, month)

    # Extract the rows relating to restrictive intervention, binned by
    # provider.
    key = 'Provider; Restrictive Intervention Type'
    df_RI = df_input[df_input.BREAKDOWN == key]

    if len(df_RI) == 0:

        key = 'Provider; Restrictive intervention type'
        df_RI = df_input[df_input.BREAKDOWN == key]


    # Filter again by the metric that we are interested in, in this case
    # metric MHS76 'Number of people subject to restrictive intervention'.
    metric_key = 'MHS76'
    df_RI_metric = df_RI[df_RI.METRIC == metric_key]

    # Loop through these rows, identifying the intervention type and
    # provider and storing in the appropriate entry of the output
    # dictionary of data frames.
    # Pandas has a rather ugly syntax for doing this.
    for row in df_RI_metric.itertuples(index = True, name = 'Pandas'):
        
        provider = row.LEVEL_ONE
        intervention = row.LEVEL_TWO
        value = row.METRIC_VALUE

        # In the CSV files, the intervention can be written as an
        # integer string ('1', '2', ...), a zero-padded integer string
        # ('01', '02', ...) or as 'UNKNOWN', so (as before), we need to
        # standardise the integer values.
        if intervention != 'UNKNOWN':

            # Same as before: cast to int, then format as zero-padded
            # string.
            intervention = '{:02d}'.format(int(intervention))

        # Parse the value from the table, treating missing values as zero
        # and converting from string to integer.
        if value == '*':

            value = 0

        else:

            value = int(value)

        # Store the value in the appropriate entry in our output array.
        restraint_data.loc[intervention, year_month_str, provider] = value 

    return restraint_data

def parse_restraint_csv_files(dir_input, dir_output):

    # Specify directory name, the expect format of the CSV file names, and
    # the breakpoints in the CSV file names where the year and month can
    # be found.
    dir_data = os.path.join(dir_input, 'restraint')
                        #'Restraint Data Oct 2021 to September 2022_tidied')
    #file_name_format = 'MHSDS Data_Rstr_{:}Prf_{:d}_final.csv'
    file_name_format = 'MHSDS Data_Rstr_{:}Prf_{:d}.csv'
    file_str_break_points = [16, 23]
    print('\nParsing CSV files in restraint folder: {:}'.format(dir_data))

    # Check whether the output already exists. In this case, parsing will be
    # skipped.
    output_name = 'restraint_data'
    file_output_nc = '{:}.nc'.format(output_name)
    path_output_nc = os.path.join(dir_output, file_output_nc)
    if os.path.exists(path_output_nc):

        print('Output file {:} already exists. Skipping processing'.format(
            path_output_nc))
        return

    # Search the input directory to find an array listing the sorted year-month
    # pairs.
    year_month_pairs = \
        get_list_of_input_year_month_pairs(dir_data, file_str_break_points)

    # Loop through the CSV files and find a list of all the intervention types
    # and healthcare providers.
    providers, interventions, providers_description, interventions_description\
        = get_list_of_providers_and_interventions_from_restraint_csv(dir_data,
                year_month_pairs, file_name_format)

    # Prepare the output array for the restraint data (see the function
    # for more information about the output array).
    restraint_data = prepare_restraint_data_output_array(year_month_pairs,
                        interventions, providers,
                        interventions_description, providers_description)

    # Loop over the reporting period a second time to store the data of
    # interest in the output array.
    # Note that looping twice requires the CSV files to be loaded twice,
    # which is slower. This could be avoided by keeping the files in memory
    # (although this risks using up too much memory) or by combining the
    # two loops (but this would be less elegant and harder to follow).
    for i in range(restraint_data.sizes['year_month']):
        
        # For this reporting period, parse the CSV file and update the
        # output array with the results.
        restraint_data = parse_one_restraint_csv_file(dir_data,
                            file_name_format,
                            *year_month_pairs[i, :],
                            restraint_data)

    print("Saving to {:}".format(path_output_nc))
    restraint_data.to_netcdf(path_output_nc)

    return

# --- Handling main database CSV files. ----------------------------------------
def get_list_of_providers_from_main_database_csv(dir_data, year_month_pairs, file_name_format):

    # Loop over the year-month pairs to get a list of providers.
    # We also store the descriptions (e.g. provider 'RXM' has description
    # 'DERBYSHIRE HEALTHCARE NHS FOUNDATION TRUST'.)
    #
    # Prepare the empty lists.
    providers_list = []
    providers_description_list = []
    #
    # Loop over the number of monthly reporting periods.
    n_monthly_reports = year_month_pairs.shape[0]
    for i in range(n_monthly_reports):

        # Load the input file for this year-month pair and store it in a
        # Pandas data frame.
        df_input = load_csv_from_year_month(dir_data,
                                            file_name_format,
                                            *year_month_pairs[i, :])

        # Get a list of all providers mentioned in this file, and append to the
        # master list.
        # In Pandas, if the dataframe has a column e.g. 'BREAKDOWN', it can
        # be accessed as follows: dataframe.BREAKDOWN
        key = 'Provider'
        df_providers = df_input[df_input.BREAKDOWN == key]
        providers_list.extend(list(df_providers.PRIMARY_LEVEL))
        providers_description_list.extend(list(df_providers.PRIMARY_LEVEL_DESCRIPTION))

    # Reduce the lists of providers and interventions to unique, sorted lists.
    providers, providers_description = \
        reduce_to_unique_sorted_list(providers_list, providers_description_list)

    # Report number of providers and interventions identified.
    n_providers = len(providers)
    print('Found {:d} providers'.format(n_providers))

    return providers, providers_description

def prepare_bed_days_output_array(year_month_pairs, providers, providers_description):

    # Create a list of strings which combine the year-month pairs to act as an index.
    year_month_str_list = [year_month_fmt.format(*year_month) for year_month
                                in year_month_pairs]

    # Get the dimensions of the output array.
    n_providers = len(providers)
    n_monthly_reports = len(year_month_str_list)

    # Prepare the output, an xarray DataArray.
    # It has two dimensions: provider and month.
    # So, for example, if the month is October 2021, and the provider is DE8
    # (Elysium Healthcare), then the value (bed days in reporting period)
    # can be accessed with this syntax 
    #   n_bed_days = bed_days_data.sel( year_month = '2021_10',
    #                                   provider = 'DE8')
    # and it can be updated with this syntax:
    #   bed_days_data.loc['2021_10', 'DE8'] = n_bed_days
    # See the function print_summaries() in plot_results.py for more
    # examples of manipulating the DataArray.
    empty_array = np.zeros((n_monthly_reports, n_providers),
                        dtype = int)
    bed_days_data = xr.DataArray(empty_array,
            coords = [year_month_str_list, list(providers)],
            dims = ['year_month', 'provider'])

    # Set the attributes of the data array (these are just labels, but can
    # help to make data more readable).
    # Note: These may not be saved properly if the netCDF Python library isn't
    # installed.
    bed_days_data.attrs["long_name"] = 'Bed days'
    bed_days_data.attrs["units"] = "bed days"
    bed_days_data.attrs["description"] = 'Metric MHS24 (bed days in '\
        'reporting period gridded by provider and monthly reporting period.'

    # We store the full variable names as strings, because they are useful later
    # when inspecting the data.
    # (They should be dictionaries but saving dictionary attributes of NetCDF
    # files is not supported; they can be converted back to dicts when loading
    # the files later.)
    bed_days_data.provider.attrs['description'] = str({int_ : int_desc 
            for int_, int_desc in zip(providers, providers_description)})

    return bed_days_data

def parse_main_database_csv_files(dir_input, dir_output):

    # Specify directory name, the expect format of the CSV file names, and
    # the breakpoints in the CSV file names where the year and month can
    # be found.
    dir_data = os.path.join(dir_input, 'main_database') 
    #                    'Main database Oct 2021 to September 2022_tidied')
    #file_name_format = 'MHSDS Data_{:}Prf_{:d}_final.csv'
    file_name_format = 'MHSDS Data_{:}Prf_{:d}.csv'
    file_str_break_points = [11, 18]
    print('\nParsing CSV files in main database folder: {:}'.format(dir_data))

    # Check whether the output already exists. In this case, parsing will be
    # skipped.
    output_name = 'bed_days_data'
    file_output_nc = '{:}.nc'.format(output_name)
    path_output_nc = os.path.join(dir_output, file_output_nc)
    if os.path.exists(path_output_nc):

        print('Output file {:} already exists. Skipping processing'.format(
            path_output_nc))

        return

    # Search the input directory to find an array listing the sorted year-month
    # pairs.
    year_month_pairs = \
        get_list_of_input_year_month_pairs(dir_data, file_str_break_points)

    # Loop through the CSV files and find a list of all the healthcare
    # providers.
    providers, providers_description = \
            get_list_of_providers_from_main_database_csv(
                dir_data, year_month_pairs, file_name_format)

    # Prepare the output array for the data from the main database CSV files
    # (see the function for more information about the output array).
    bed_days_data = prepare_bed_days_output_array(year_month_pairs,
                        providers, providers_description)

    # Loop over the reporting period a second time to store the data of
    # interest in the output array.
    # Note that looping twice requires the CSV files to be loaded twice,
    # which is slower. This could be avoided by keeping the files in memory
    # (although this risks using up too much memory) or by combining the
    # two loops (but this would be less elegant and harder to follow).
    for i in range(bed_days_data.sizes['year_month']):
        
        # For this reporting period, parse the CSV file and update the
        # output array with the results.
        bed_days_data = parse_one_main_database_csv_file(dir_data,
                            file_name_format,
                            *year_month_pairs[i, :],
                            bed_days_data)

    print("Saving to {:}".format(path_output_nc))
    bed_days_data.to_netcdf(path_output_nc)

    return

def parse_one_main_database_csv_file(dir_data, file_name_format, year, month, bed_days_data):

    # Load the input file for this year-month pair and store it in a
    # Pandas data frame.
    df_input = load_csv_from_year_month(dir_data, file_name_format, year, month)

    # Get the year-month index.
    year_month_str = year_month_fmt.format(year, month)

    # Extract the rows which are binned by provider.
    key = 'Provider'
    df_provider = df_input[df_input.BREAKDOWN == key]

    # Filter again by the metric that we are interested in, in this case
    # metric MHS24 'Bed days in RP' (RP = reporting period).
    metric_key = 'MHS24'
    df_metric = df_provider[df_provider.MEASURE_ID == metric_key]

    # Loop through these rows, identifying the intervention type and
    # provider and storing in the appropriate entry of the output
    # dictionary of data frames.
    # Pandas has a rather ugly syntax for doing this.
    for row in df_metric.itertuples(index = True, name = 'Pandas'):
        
        provider = row.PRIMARY_LEVEL
        value = row.MEASURE_VALUE

        # Parse the value from the table, treating missing values as zero
        # and converting from string to integer.
        if value == '*':

            value = 0

        else:

            value = int(value)

        # Store the value in the appropriate entry in our output array.
        bed_days_data.loc[year_month_str, provider] = value 

    return bed_days_data

def main():

    # Define some file and directory paths.
    # We use the 'os' module to create the file path string from its components.
    #
    # Define path (relative to this file) to the directory where the input
    # files are stored. '../' means go down one directory level.
    dir_input = '../new_data'
    dir_output = 'output'
    assert os.path.isdir(dir_output), 'The output directory ({:}) does not '\
            'exist. Please create it.'.format(dir_output)
    
    # Read the CSV files (main database and restraint data) and extract the
    # data that we are interested in.
    parse_main_database_csv_files(dir_input, dir_output)
    parse_restraint_csv_files(dir_input, dir_output)

    return

# 'Main sentinel', so code can be run from command line.
if __name__ == '__main__':

    main()
