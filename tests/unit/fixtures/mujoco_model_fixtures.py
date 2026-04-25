"""This module contains functions to create MuJoCoModels for use in tests."""

import numpy as np

# noinspection PyProtectedMember
from pterasoftware import _mujoco_model


def make_basic_mujoco_model_fixture():
    """This method makes a fixture that is a MuJoCoModel with basic parameters
    representing a simple rigid body at rest with identity orientation.

    :return basic_mujoco_model_fixture: MuJoCoModel
        This is the MuJoCoModel with basic parameters.
    """
    basic_mujoco_model_fixture = _mujoco_model.MuJoCoModel(
        name="test_airplane",
        weight=9.80665,
        omegas_BP1__E=(0.0, 0.0, 0.0),
        g_E=(0.0, 0.0, 9.80665),
        T_pas_BP1_CgP1_to_E_CgP1=np.eye(4, dtype=float),
        vCg_E__E=(10.0, 0.0, 0.0),
        I_BP1_CgP1=np.eye(3, dtype=float),
        delta_time=0.01,
    )

    return basic_mujoco_model_fixture


def make_rotated_mujoco_model_fixture():
    """This method makes a fixture that is a MuJoCoModel with a 90 degree rotation
    about the z axis and non zero initial angular velocity.

    :return rotated_mujoco_model_fixture: MuJoCoModel
        This is the MuJoCoModel with rotated initial orientation.
    """
    T_pas = np.eye(4, dtype=float)
    T_pas[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )

    rotated_mujoco_model_fixture = _mujoco_model.MuJoCoModel(
        name="rotated_airplane",
        weight=19.6133,
        omegas_BP1__E=(0.0, 0.0, 10.0),
        g_E=(0.0, 0.0, 9.80665),
        T_pas_BP1_CgP1_to_E_CgP1=T_pas,
        vCg_E__E=(0.0, 5.0, -1.0),
        I_BP1_CgP1=np.diag([2.0, 3.0, 4.0]),
        delta_time=0.005,
    )

    return rotated_mujoco_model_fixture


def make_basic_mujoco_model_name_fixture():
    """This method makes a fixture that is the name used by the basic MuJoCoModel
    fixture.

    :return: str
        The name of the basic MuJoCoModel fixture.
    """
    return "test_airplane"


def make_basic_mujoco_model_weight_fixture():
    """This method makes a fixture that is the weight used by the basic MuJoCoModel
    fixture.

    :return: float
        The weight in Newtons.
    """
    return 9.80665


def make_basic_mujoco_model_delta_time_fixture():
    """This method makes a fixture that is the delta_time used by the basic MuJoCoModel
    fixture.

    :return: float
        The time step in seconds.
    """
    return 0.01
