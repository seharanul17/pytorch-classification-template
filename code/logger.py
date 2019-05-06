import os
import sys
import datetime
import logging
from config import config

log_path = os.path.join(config['@root'], config['@save'], config['@log'])
log_name = os.path.join(log_path, config['version']+'.log')
if not os.path.exists(log_path): os.makedirs(log_path)
if not os.path.exists(log_name): 
    with open (log_name, 'a') : 
        os.utime(log_name, None)

logger = logging.getLogger(config['version'])
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(log_name) #file write
streamHandler = logging.StreamHandler() #cmd show
formatter = logging.Formatter('%(levelname)s | %(message)s')
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

current_time = datetime.datetime.now()
logger.info('')
logger.info('')
logger.info(current_time.strftime("%Y.%m.%d - %Hh:%Mm:%Ss"))
logger.info(str(' '.join(sys.argv)))
logger.info(' -------------------- setting --------------------')
logger.info('')
for key in config:
    logger.info('   {} : {}'.format(key, config[key]))
logger.info('')
logger.info(' -------------------- setting --------------------')