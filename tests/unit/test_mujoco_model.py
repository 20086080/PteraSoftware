"""This module contains classes to test MuJoCoModels."""

import struct
import unittest

import mujoco
import numpy as np
import numpy.testing as npt

# noinspection PyProtectedMember
from pterasoftware import _mujoco_model
from tests.unit.fixtures import mujoco_model_fixtures


class TestMuJoCoModelInit(unittest.TestCase):
    """This class contains methods for testing MuJoCoModel initialization."""

    def setUp(self):
        """Set up test fixtures for MuJoCoModel initialization tests."""
        self.model = mujoco_model_fixtures.make_basic_mujoco_model_fixture()

    def test_instantiation_returns_correct_type(self):
        """Test that MuJoCoModel instantiation returns a MuJoCoModel."""
        self.assertIsInstance(self.model, _mujoco_model.MuJoCoModel)

    def test_xml_str_contains_model_name(self):
        """Test that the generated XML contains the Airplane name."""
        name = mujoco_model_fixtures.make_basic_mujoco_model_name_fixture()
        self.assertIn(name, self.model.xml_str)

    def test_xml_str_contains_timestep(self):
        """Test that the generated XML contains the delta_time."""
        delta_time = mujoco_model_fixtures.make_basic_mujoco_model_delta_time_fixture()
        self.assertIn(str(delta_time), self.model.xml_str)

    def test_model_is_mj_model(self):
        """Test that the model property returns an MjModel."""
        self.assertIsInstance(self.model.model, mujoco.MjModel)

    def test_data_is_mj_data(self):
        """Test that the data property returns an MjData."""
        self.assertIsInstance(self.model.data, mujoco.MjData)

    def test_body_id_is_int(self):
        """Test that body_id is an int."""
        self.assertIsInstance(self.model.body_id, int)

    def test_initial_key_frame_id_is_int(self):
        """Test that initial_key_frame_id is an int."""
        self.assertIsInstance(self.model.initial_key_frame_id, int)

    def test_initial_qpos_shape(self):
        """Test that initial_qpos has the correct shape for a free joint."""
        self.assertEqual(self.model.initial_qpos.shape, (7,))

    def test_initial_qvel_shape(self):
        """Test that initial_qvel has the correct shape for a free joint."""
        self.assertEqual(self.model.initial_qvel.shape, (6,))

    def test_initial_qvel_contains_velocity(self):
        """Test that initial_qvel contains the initial velocity."""
        npt.assert_allclose(self.model.initial_qvel[0:3], [10.0, 0.0, 0.0], atol=1e-14)

    def test_initial_qvel_contains_angular_velocity(self):
        """Test that initial_qvel contains the initial angular velocity in radians per
        second.
        """
        npt.assert_allclose(self.model.initial_qvel[3:6], [0.0, 0.0, 0.0], atol=1e-14)

    def test_mass_computed_from_weight_and_gravity(self):
        """Test that the MuJoCo body mass equals weight divided by gravity magnitude."""
        weight = mujoco_model_fixtures.make_basic_mujoco_model_weight_fixture()
        g_mag = 9.80665
        expected_mass = weight / g_mag

        body_id = self.model.body_id
        actual_mass = self.model.model.body_mass[body_id]
        self.assertAlmostEqual(actual_mass, expected_mass, places=10)

    def test_timestep_set_on_model(self):
        """Test that the MuJoCo model timestep matches delta_time."""
        delta_time = mujoco_model_fixtures.make_basic_mujoco_model_delta_time_fixture()
        self.assertAlmostEqual(self.model.model.opt.timestep, delta_time, places=14)

    def test_gravity_disabled_in_mujoco(self):
        """Test that gravity is set to zero in the MuJoCo model."""
        npt.assert_array_equal(self.model.model.opt.gravity, [0.0, 0.0, 0.0])

    def test_accepts_list_inputs(self):
        """Test that MuJoCoModel accepts list inputs for array parameters."""
        model = _mujoco_model.MuJoCoModel(
            name="list_test",
            weight=9.80665,
            omegas_BP1__E=[0.0, 0.0, 0.0],
            g_E=[0.0, 0.0, 9.80665],
            T_pas_BP1_CgP1_to_E_CgP1=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            vCg_E__E=[10.0, 0.0, 0.0],
            I_BP1_CgP1=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            delta_time=0.01,
        )
        self.assertIsInstance(model, _mujoco_model.MuJoCoModel)

    def test_accepts_tuple_inputs(self):
        """Test that MuJoCoModel accepts tuple inputs for array parameters."""
        model = _mujoco_model.MuJoCoModel(
            name="tuple_test",
            weight=9.80665,
            omegas_BP1__E=(0.0, 0.0, 0.0),
            g_E=(0.0, 0.0, 9.80665),
            T_pas_BP1_CgP1_to_E_CgP1=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            vCg_E__E=(10.0, 0.0, 0.0),
            I_BP1_CgP1=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            delta_time=0.01,
        )
        self.assertIsInstance(model, _mujoco_model.MuJoCoModel)

    def test_accepts_extra_xml(self):
        """Test that MuJoCoModel accepts extra_xml and injects it into the XML."""
        extra_xml = {"visual": '<visual><quality shadowsize="2048"/></visual>'}
        model = _mujoco_model.MuJoCoModel(
            name="extra_xml_test",
            weight=9.80665,
            omegas_BP1__E=(0.0, 0.0, 0.0),
            g_E=(0.0, 0.0, 9.80665),
            T_pas_BP1_CgP1_to_E_CgP1=np.eye(4, dtype=float),
            vCg_E__E=(10.0, 0.0, 0.0),
            I_BP1_CgP1=np.eye(3, dtype=float),
            delta_time=0.01,
            extra_xml=extra_xml,
        )
        self.assertIn("shadowsize", model.xml_str)

    def test_accepts_mujoco_assets(self):
        """Test that MuJoCoModel accepts mujoco_assets without error."""
        header = b"\x00" * 80
        verts = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        triangles = b""
        for i0, i1, i2 in faces:
            normal = struct.pack("<fff", 0.0, 0.0, 0.0)
            tri = b"".join(struct.pack("<fff", *verts[i]) for i in (i0, i1, i2))
            triangles += normal + tri + struct.pack("<H", 0)
        stl_bytes = header + struct.pack("<I", len(faces)) + triangles

        mujoco_assets = {"dummy.stl": stl_bytes}
        extra_xml = {
            "asset": '<asset><mesh name="dummy" file="dummy.stl"/></asset>',
            "body": '<geom type="mesh" mesh="dummy"/>',
        }
        model = _mujoco_model.MuJoCoModel(
            name="assets_test",
            weight=9.80665,
            omegas_BP1__E=(0.0, 0.0, 0.0),
            g_E=(0.0, 0.0, 9.80665),
            T_pas_BP1_CgP1_to_E_CgP1=np.eye(4, dtype=float),
            vCg_E__E=(10.0, 0.0, 0.0),
            I_BP1_CgP1=np.eye(3, dtype=float),
            delta_time=0.01,
            extra_xml=extra_xml,
            mujoco_assets=mujoco_assets,
        )
        self.assertIsInstance(model, _mujoco_model.MuJoCoModel)

    def test_rotated_initial_orientation(self):
        """Test that a rotated initial orientation produces correct initial state."""
        model = mujoco_model_fixtures.make_rotated_mujoco_model_fixture()
        state = model.get_state()

        R = state["R_pas_E_to_BP1"]
        expected_R = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
        )
        npt.assert_allclose(R, expected_R, atol=1e-10)

    def test_non_zero_initial_angular_velocity(self):
        """Test that non zero initial angular velocity is stored correctly."""
        model = mujoco_model_fixtures.make_rotated_mujoco_model_fixture()
        state = model.get_state()
        npt.assert_allclose(state["omegas_BP1__E"], [0.0, 0.0, 10.0], atol=1e-10)

    def test_invalid_omegas_raises_type_error(self):
        """Test that invalid omegas_BP1__E raises TypeError."""
        with self.assertRaises(TypeError):
            _mujoco_model.MuJoCoModel(
                name="test",
                weight=9.80665,
                omegas_BP1__E="invalid",
                g_E=(0.0, 0.0, 9.80665),
                T_pas_BP1_CgP1_to_E_CgP1=np.eye(4, dtype=float),
                vCg_E__E=(10.0, 0.0, 0.0),
                I_BP1_CgP1=np.eye(3, dtype=float),
                delta_time=0.01,
            )

    def test_invalid_I_BP1_CgP1_shape_raises_value_error(self):
        """Test that wrong shaped I_BP1_CgP1 raises ValueError."""
        with self.assertRaises(ValueError):
            _mujoco_model.MuJoCoModel(
                name="test",
                weight=9.80665,
                omegas_BP1__E=(0.0, 0.0, 0.0),
                g_E=(0.0, 0.0, 9.80665),
                T_pas_BP1_CgP1_to_E_CgP1=np.eye(4, dtype=float),
                vCg_E__E=(10.0, 0.0, 0.0),
                I_BP1_CgP1=np.eye(4, dtype=float),
                delta_time=0.01,
            )

    def test_invalid_T_pas_shape_raises_value_error(self):
        """Test that wrong shaped T_pas_BP1_CgP1_to_E_CgP1 raises ValueError."""
        with self.assertRaises(ValueError):
            _mujoco_model.MuJoCoModel(
                name="test",
                weight=9.80665,
                omegas_BP1__E=(0.0, 0.0, 0.0),
                g_E=(0.0, 0.0, 9.80665),
                T_pas_BP1_CgP1_to_E_CgP1=np.eye(3, dtype=float),
                vCg_E__E=(10.0, 0.0, 0.0),
                I_BP1_CgP1=np.eye(3, dtype=float),
                delta_time=0.01,
            )

    def test_symmetrizes_inertia_matrix(self):
        """Test that an asymmetric inertia matrix is symmetrized."""
        I_asymmetric = np.array(
            [[1.0, 0.2, 0.0], [0.4, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=float
        )
        model = _mujoco_model.MuJoCoModel(
            name="sym_test",
            weight=9.80665,
            omegas_BP1__E=(0.0, 0.0, 0.0),
            g_E=(0.0, 0.0, 9.80665),
            T_pas_BP1_CgP1_to_E_CgP1=np.eye(4, dtype=float),
            vCg_E__E=(10.0, 0.0, 0.0),
            I_BP1_CgP1=I_asymmetric,
            delta_time=0.01,
        )
        expected_Ixy = (0.2 + 0.4) / 2
        self.assertIn(str(expected_Ixy), model.xml_str)


class TestMuJoCoModelImmutability(unittest.TestCase):
    """This class contains methods for testing MuJoCoModel immutability patterns."""

    def setUp(self):
        """Set up test fixtures for MuJoCoModel immutability tests."""
        self.model = mujoco_model_fixtures.make_basic_mujoco_model_fixture()

    def test_immutable_xml_str_raises_attribute_error(self):
        """Test that setting xml_str raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.xml_str = "new xml"

    def test_immutable_model_raises_attribute_error(self):
        """Test that setting model raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.model = None

    def test_immutable_body_id_raises_attribute_error(self):
        """Test that setting body_id raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.body_id = 99

    def test_immutable_initial_key_frame_id_raises_attribute_error(self):
        """Test that setting initial_key_frame_id raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.initial_key_frame_id = 99

    def test_immutable_initial_qpos_raises_attribute_error(self):
        """Test that setting initial_qpos raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.initial_qpos = np.zeros(7)

    def test_immutable_initial_qvel_raises_attribute_error(self):
        """Test that setting initial_qvel raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.initial_qvel = np.zeros(6)

    def test_immutable_data_raises_attribute_error(self):
        """Test that setting data raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.model.data = None

    def test_initial_qpos_is_read_only_array(self):
        """Test that initial_qpos array is not writeable."""
        with self.assertRaises(ValueError):
            self.model.initial_qpos[0] = 999.0

    def test_initial_qvel_is_read_only_array(self):
        """Test that initial_qvel array is not writeable."""
        with self.assertRaises(ValueError):
            self.model.initial_qvel[0] = 999.0


class TestMuJoCoModelApplyLoads(unittest.TestCase):
    """This class contains methods for testing MuJoCoModel.apply_loads."""

    def setUp(self):
        """Set up test fixtures for apply_loads tests."""
        self.model = mujoco_model_fixtures.make_basic_mujoco_model_fixture()

    def test_apply_loads_sets_forces(self):
        """Test that apply_loads sets the force on the body."""
        forces_E = np.array([1.0, 2.0, 3.0])
        moments_E_CgP1 = np.array([0.0, 0.0, 0.0])
        self.model.apply_loads(forces_E, moments_E_CgP1)

        applied = self.model.data.xfrc_applied[self.model.body_id]
        npt.assert_allclose(applied[0:3], forces_E, atol=1e-14)

    def test_apply_loads_sets_moments(self):
        """Test that apply_loads sets the moment on the body."""
        forces_E = np.array([0.0, 0.0, 0.0])
        moments_E_CgP1 = np.array([4.0, 5.0, 6.0])
        self.model.apply_loads(forces_E, moments_E_CgP1)

        applied = self.model.data.xfrc_applied[self.model.body_id]
        npt.assert_allclose(applied[3:6], moments_E_CgP1, atol=1e-14)

    def test_apply_loads_accepts_lists(self):
        """Test that apply_loads accepts list inputs."""
        self.model.apply_loads([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        applied = self.model.data.xfrc_applied[self.model.body_id]
        self.assertAlmostEqual(applied[0], 1.0, places=14)

    def test_apply_loads_accepts_tuples(self):
        """Test that apply_loads accepts tuple inputs."""
        self.model.apply_loads((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        applied = self.model.data.xfrc_applied[self.model.body_id]
        self.assertAlmostEqual(applied[0], 1.0, places=14)

    def test_apply_loads_overwrites_previous(self):
        """Test that apply_loads overwrites previously applied loads."""
        self.model.apply_loads([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.model.apply_loads([10.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        applied = self.model.data.xfrc_applied[self.model.body_id]
        npt.assert_allclose(applied[0:3], [10.0, 0.0, 0.0], atol=1e-14)
        npt.assert_allclose(applied[3:6], [0.0, 0.0, 0.0], atol=1e-14)

    def test_apply_loads_invalid_forces_raises_type_error(self):
        """Test that invalid forces input raises TypeError."""
        with self.assertRaises(TypeError):
            self.model.apply_loads("invalid", [0.0, 0.0, 0.0])

    def test_apply_loads_invalid_moments_raises_type_error(self):
        """Test that invalid moments input raises TypeError."""
        with self.assertRaises(TypeError):
            self.model.apply_loads([0.0, 0.0, 0.0], "invalid")


class TestMuJoCoModelStep(unittest.TestCase):
    """This class contains methods for testing MuJoCoModel.step."""

    def setUp(self):
        """Set up test fixtures for step tests."""
        self.model = mujoco_model_fixtures.make_basic_mujoco_model_fixture()

    def test_step_advances_time(self):
        """Test that step advances the simulation time by delta_time."""
        delta_time = mujoco_model_fixtures.make_basic_mujoco_model_delta_time_fixture()
        self.model.step()
        self.assertAlmostEqual(self.model.data.time, delta_time, places=14)

    def test_multiple_steps_advance_time(self):
        """Test that multiple steps advance time correctly."""
        delta_time = mujoco_model_fixtures.make_basic_mujoco_model_delta_time_fixture()
        num_steps = 10
        for _ in range(num_steps):
            self.model.step()
        self.assertAlmostEqual(self.model.data.time, num_steps * delta_time, places=10)

    def test_step_with_force_changes_velocity(self):
        """Test that stepping with an applied force changes the velocity."""
        initial_state = self.model.get_state()
        initial_velocity = initial_state["velocity_E__E"].copy()

        self.model.apply_loads([100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.model.step()

        final_state = self.model.get_state()
        final_velocity = final_state["velocity_E__E"]

        self.assertFalse(np.allclose(initial_velocity, final_velocity))


class TestMuJoCoModelGetState(unittest.TestCase):
    """This class contains methods for testing MuJoCoModel.get_state."""

    def setUp(self):
        """Set up test fixtures for get_state tests."""
        self.model = mujoco_model_fixtures.make_basic_mujoco_model_fixture()

    def test_get_state_returns_dict(self):
        """Test that get_state returns a dict."""
        state = self.model.get_state()
        self.assertIsInstance(state, dict)

    def test_get_state_contains_expected_keys(self):
        """Test that get_state returns all expected keys."""
        state = self.model.get_state()
        expected_keys = {
            "position_E_E",
            "R_pas_E_to_BP1",
            "velocity_E__E",
            "omegas_BP1__E",
            "time",
        }
        self.assertEqual(set(state.keys()), expected_keys)

    def test_position_shape(self):
        """Test that position_E_E has shape (3,)."""
        state = self.model.get_state()
        self.assertEqual(state["position_E_E"].shape, (3,))

    def test_rotation_matrix_shape(self):
        """Test that R_pas_E_to_BP1 has shape (3,3)."""
        state = self.model.get_state()
        self.assertEqual(state["R_pas_E_to_BP1"].shape, (3, 3))

    def test_velocity_shape(self):
        """Test that velocity_E__E has shape (3,)."""
        state = self.model.get_state()
        self.assertEqual(state["velocity_E__E"].shape, (3,))

    def test_omegas_shape(self):
        """Test that omegas_BP1__E has shape (3,)."""
        state = self.model.get_state()
        self.assertEqual(state["omegas_BP1__E"].shape, (3,))

    def test_time_is_float(self):
        """Test that time is a float."""
        state = self.model.get_state()
        self.assertIsInstance(state["time"], float)

    def test_initial_position_at_origin(self):
        """Test that the initial position is at the origin."""
        state = self.model.get_state()
        npt.assert_allclose(state["position_E_E"], [0.0, 0.0, 0.0], atol=1e-14)

    def test_initial_time_is_zero(self):
        """Test that the initial time is zero."""
        state = self.model.get_state()
        self.assertAlmostEqual(state["time"], 0.0, places=14)

    def test_identity_rotation_at_init(self):
        """Test that identity T_pas produces identity R_pas_E_to_BP1."""
        state = self.model.get_state()
        npt.assert_allclose(state["R_pas_E_to_BP1"], np.eye(3), atol=1e-10)

    def test_initial_velocity_matches_input(self):
        """Test that the initial velocity matches the input vCg_E__E."""
        state = self.model.get_state()
        npt.assert_allclose(state["velocity_E__E"], [10.0, 0.0, 0.0], atol=1e-14)

    def test_get_state_returns_copies(self):
        """Test that get_state returns copies that do not alias internal data."""
        state1 = self.model.get_state()
        state1["position_E_E"][0] = 999.0
        state2 = self.model.get_state()
        self.assertNotAlmostEqual(state2["position_E_E"][0], 999.0)


class TestMuJoCoModelReset(unittest.TestCase):
    """This class contains methods for testing MuJoCoModel.reset."""

    def setUp(self):
        """Set up test fixtures for reset tests."""
        self.model = mujoco_model_fixtures.make_basic_mujoco_model_fixture()

    def test_reset_restores_time_to_zero(self):
        """Test that reset restores time to zero."""
        self.model.step()
        self.model.step()
        self.model.reset()
        self.assertAlmostEqual(self.model.data.time, 0.0, places=14)

    def test_reset_restores_initial_qpos(self):
        """Test that reset restores initial generalized positions."""
        initial_qpos = self.model.initial_qpos.copy()
        self.model.apply_loads([100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for _ in range(10):
            self.model.step()
        self.model.reset()
        npt.assert_allclose(self.model.data.qpos, initial_qpos, atol=1e-14)

    def test_reset_restores_initial_qvel(self):
        """Test that reset restores initial generalized velocities."""
        initial_qvel = self.model.initial_qvel.copy()
        self.model.apply_loads([100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for _ in range(10):
            self.model.step()
        self.model.reset()
        npt.assert_allclose(self.model.data.qvel, initial_qvel, atol=1e-14)

    def test_reset_clears_applied_loads(self):
        """Test that reset clears any applied loads."""
        self.model.apply_loads([100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        self.model.reset()
        applied = self.model.data.xfrc_applied[self.model.body_id]
        npt.assert_array_equal(applied, np.zeros(6))

    def test_reset_produces_same_state_as_init(self):
        """Test that reset produces the same get_state output as after init."""
        initial_state = self.model.get_state()

        self.model.apply_loads([100.0, 0.0, 0.0], [0.0, 0.0, 10.0])
        for _ in range(20):
            self.model.step()

        self.model.reset()
        reset_state = self.model.get_state()

        npt.assert_allclose(
            reset_state["position_E_E"], initial_state["position_E_E"], atol=1e-14
        )
        npt.assert_allclose(
            reset_state["velocity_E__E"], initial_state["velocity_E__E"], atol=1e-14
        )
        npt.assert_allclose(
            reset_state["omegas_BP1__E"], initial_state["omegas_BP1__E"], atol=1e-14
        )
        npt.assert_allclose(
            reset_state["R_pas_E_to_BP1"],
            initial_state["R_pas_E_to_BP1"],
            atol=1e-10,
        )
        self.assertAlmostEqual(reset_state["time"], initial_state["time"], places=14)


if __name__ == "__main__":
    unittest.main()
