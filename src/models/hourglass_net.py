"""MnistNet architecture."""

from typing import Dict

import tensorflow as tf
from core import BaseDataSource, BaseModel
from .hourglass_module import hourglass, residual # I deleted my helper files (ping me if you need anything (tho you shouldn't))
from util.gaussian import gaussian_maps


import logging
logger = logging.getLogger(__name__)

class StackedHourglass(BaseModel):
    """Hourglass model [Newell et al. 2016] """
    #######   XANDER I AM ASSUMING THAT FOR YOUR TASK YOU HAVE A SIMILAR ARCHITECTURE, HOWEVER IF IT'S NOT LIKE THAT
    #       JUST SAY THAT YOU "ONLY NEED" TO PUT THE MODEL HERE, IN A SIMILAR FASHION TO MINE, AND THEN SELECT IN THE
    #       MAIN FILE (src/main.py) THE ELEMENTS YOU WANT TO TRAIN AND WHICH OF THE LOSSES (DEFINED HERE AT THE BOTTOM)
    #       YOU WANT TO USE. AFTER THAT THE BASEMODEL WILL TAKE THE REST OF EVERYTHING.... WELL, IDK WHY I'M TELLING YOU 
    #       THIS AS YOU ALREADY KNOW....  ¯\_(ツ)_/¯
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        logger.info('Start building model {}'.format(__name__))
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors  # Data source automatically handles the datafiles
        x = input_tensors['img']    # Inputs
        y = input_tensors['kp_2D']  # Targets
        logging.debug(y)

        filters = 256
        num_joints = 21  # TODO: Why can't I use "tf.shape(y[2])"?
        hourglass_stacks = 8
        decrease_factor = 2
        logger.debug('Inputs - {} \nTargets - {}'.format(x,y))
        y = y / decrease_factor  # TODO: Change the way we rescale things
        outs = []

        with tf.variable_scope('pre'):
            x = tf.layers.conv2d(x, filters=filters/4, kernel_size=7, strides=decrease_factor,
                                 padding='same', data_format='channels_first')
            x = tf.nn.relu(tf.layers.batch_normalization(x))

            x = residual(x, filters/2)
            x = residual(x, filters)

        for i in range(hourglass_stacks):
            with tf.variable_scope('hourglass_{}'.format(i)):
                x = hourglass(x)

                r = residual(x, filters)
                logger.debug(r)
                r = tf.layers.conv2d(r, filters, 1, data_format='channels_first')
                logger.debug(r)
                r = tf.nn.relu(tf.layers.batch_normalization(x))
                logger.debug(r)
                # compute block loss with o
                o = tf.layers.conv2d(r, num_joints, 1, data_format='channels_first')
                logger.debug('{} - Stack output'.format(o))
                outs.append(o)
                
                if i < hourglass_stacks:
                    join1 = tf.layers.conv2d(o, filters, 1, data_format='channels_first')
                    logger.debug('{} - Output branch'.format(join1))
                    join2 = tf.layers.conv2d(r, filters, 1, data_format='channels_first')
                    logger.debug('{} - Main branck'.format(join2))
                    x = join1 + join2
                    logger.debug('{} - Juntion (Add branches)'.format(x))        
        
        # Convert y (target) coords (x_T, y_T) to 2D distribution map
        y = tf.map_fn(gaussian_maps, tf.cast(y, tf.float32))  # Are they ints or floats?
        logger.debug(' ---- Loss ----\n Targets - {}\n Preds - {}'.format(y, outs))
        
        # Loss
        loss_terms = {}
        for idx, o in enumerate(outs):
            loss_terms['map_mse_{}'.format(idx)] = tf.losses.mean_squared_error(o, y) 
            
        # Define outputs
        # loss_terms = {  # To optimize  --> Original snippet. 
            # 'kp_2D_mse': (tf.reduce_mean(outs))
                            # #tf.reduce_mean(tf.squared_difference(x, y)),
        # }
        logger.info('Model {} building exiting.'.format(__name__))
        return {'kp_2D': outs[-1]}, loss_terms, {}
