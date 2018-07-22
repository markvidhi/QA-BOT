"""configuration file"""

import configparser 
config = configparser.ConfigParser()

config['d'] = {'threshold': '0.7','csv_file_path': 'sopra_dataset.csv'} #threshold value and datset path
with open('conf.ini', 'w') as configfile: #save values in config file
    config.write(configfile)
