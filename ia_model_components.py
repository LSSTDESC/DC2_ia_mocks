r"""
halotools model components for modelling central and scatellite intrinsic alignments
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from rotations import rotate_vector_collection
from rotations.mcrotations import random_perpendicular_directions, random_unit_vectors_3d
from rotations.vector_utilities import (elementwise_dot, elementwise_norm, normalized_vectors,
                                        angles_between_list_of_vectors)
from rotations.rotations3d import (vectors_between_list_of_vectors, vectors_normal_to_planes,
                                   rotation_matrices_from_angles)
from warnings import warn


__all__ = ('RandomAlignment',)
__author__ = ('Duncan Campbell',)


class RandomAlignment(object):
    """
    class to model random galaxy orientations
    """
    def __init__(self, gal_type='centrals', **kwargs):
        """
        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = []
        self._methods_to_inherit = ([])
        self.param_dict = ({})

    def assign_orientation(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
        else:
            N = kwargs['size']

        # assign random orientations
        major_v = random_unit_vectors_3d(N)
        inter_v = random_perpendicular_directions(major_v)
        minor_v = normalized_vectors(np.cross(major_v, inter_v))

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*N)
                msg = ("Because `gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v
