

def set_pycortex_config_file(data_folder):

    # Import necessary modules
    import os
    import cortex
    import pdb
    from pathlib import Path
    deb = pdb.set_trace

    # Define the new database and colormaps folder
    pycortex_db_folder = data_folder + '/derivatives/pp_data/cortex/db'
    pycortex_cm_folder = data_folder + '/derivatives/pp_data/cortex/colormaps'

    # Get pycortex config file location
    pycortex_config_file  = cortex.options.usercfg


    # Create name of new config file that will be written
    new_pycortex_config_file = pycortex_config_file[:-4] + '_new.cfg'

    # Create the new config file
    Path(new_pycortex_config_file).touch()

    # Open the config file in read mode and the newly created one in write mode.
    # Loop over every line in the original file and copy it into the new one.
    # For the lines containing either 'filestore' or 'colormap', it will
    # change the saved folder path to the newly created one above (e.g. pycortex_db_folder)
    with open(pycortex_config_file, 'r') as fileIn:
        with open(new_pycortex_config_file, 'w') as fileOut:

            for line in fileIn:

                if 'filestore' in line:
                    newline = 'filestore=' + pycortex_db_folder
                    fileOut.write(newline)
                    newline = '\n'

                elif 'colormaps' in line:
                    newline = 'colormaps=' + pycortex_cm_folder
                    fileOut.write(newline)
                    newline = '\n'

                else:
                    newline = line

                fileOut.write(newline)

    
    # Renames the original config file als '_old' and the newly created one to the original name
    os.rename(pycortex_config_file, pycortex_config_file[:-4] + '_old.cfg')
    os.rename(new_pycortex_config_file, pycortex_config_file)
    return None

