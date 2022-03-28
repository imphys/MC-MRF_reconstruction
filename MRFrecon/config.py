import configparser
import os.path
import sys


# written in function so other modules can load its contents as well
def load_config_file(filepath):
    # Load config file
    configfile = configparser.ConfigParser()
    if filepath is not None and filepath.endswith('.ini'):
        print('Read file ' + str(filepath))
        configfile.read(filepath)
    else:
        configfile.read(os.path.join(filepath, 'config.ini'))
    # add configpath to config
    configfile['DEFAULT']['configpath'] = os.path.dirname(filepath) + '/'
    return configfile


def load_settings(filepath=None, run=None):
    # Try to read path to config.ini file
    if filepath is None:
        try:
            filepath = sys.argv[1]
        except IndexError:
            filepath = r''
    # Specify what run to excecute
    if run is None:
        try:
            run = int(sys.argv[2])
        except IndexError:
            run = 1
        except ValueError:
            run = 1

    config = load_config_file(filepath)
    # get settings for runs
    run_name = config['RUN'][str(run)]
    config['DEFAULT']['path_extension'] = run_name
    settings = config[run_name]
    return settings


settings = None
