"""This module contains classes to test FreeFlightUnsteadyProblems."""

import unittest

import numpy as np

import pterasoftware as ps
from tests.unit.fixtures import problem_fixtures


class TestFreeFlightUnsteadyProblem(unittest.TestCase):
    """This is a class with functions to test FreeFlightUnsteadyProblems."""

    def setUp(self):
        """Set up a fresh FreeFlightUnsteadyProblem for each test."""
        self.problem = (
            problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()
        )

    def test_initialization_valid_parameters(self):
        """Test FreeFlightUnsteadyProblem initialization with valid parameters."""
        self.assertIsInstance(
            self.problem,
            ps.problems.FreeFlightUnsteadyProblem,
        )
        self.assertIsInstance(
            self.problem,
            ps.problems._CoupledUnsteadyProblem,
        )
        self.assertIsInstance(
            self.problem.movement,
            ps.movements.free_flight_movement.FreeFlightMovement,
        )

    def test_movement_type_validation(self):
        """Test that movement must be a FreeFlightMovement."""
        with self.assertRaises(TypeError):
            ps.problems.FreeFlightUnsteadyProblem(
                movement="not_a_movement",
                I_BP1_CgP1=np.eye(3),
            )

    def test_single_airplane_movement_validation(self):
        """Test that the FreeFlightMovement has exactly one FreeFlightAirplaneMovement."""
        movement = (
            problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()._free_flight_movement
        )
        two_airplane_movement = ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=[
                movement.airplane_movements[0],
                movement.airplane_movements[0],
            ],
            operating_point_movement=movement.operating_point_movement,
            delta_time=movement.delta_time,
            prescribed_num_steps=3,
            free_num_steps=2,
        )
        with self.assertRaises(ValueError):
            ps.problems.FreeFlightUnsteadyProblem(
                movement=two_airplane_movement,
                I_BP1_CgP1=np.eye(3),
            )

    def test_I_BP1_CgP1_symmetry_validation(self):
        """Test that I_BP1_CgP1 must be symmetric."""
        movement = (
            problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()._free_flight_movement
        )
        asymmetric_inertia = np.array(
            [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
        )
        with self.assertRaises(ValueError):
            ps.problems.FreeFlightUnsteadyProblem(
                movement=movement,
                I_BP1_CgP1=asymmetric_inertia,
            )

    def test_external_forces_fn_validation(self):
        """Test that external_forces_fn must be callable or None."""
        movement = (
            problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()._free_flight_movement
        )
        with self.assertRaises(TypeError):
            ps.problems.FreeFlightUnsteadyProblem(
                movement=movement,
                I_BP1_CgP1=np.eye(3),
                external_forces_fn="not_callable",
            )

    def test_external_forces_fn_default_none(self):
        """Test that external_forces_fn defaults to None."""
        self.assertIsNone(self.problem.external_forces_fn)

    def test_only_final_results_forced_false(self):
        """Test that only_final_results is always False for coupled problems."""
        self.assertFalse(self.problem.only_final_results)

    def test_num_steps_from_movement(self):
        """Test that num_steps is taken from the movement."""
        self.assertEqual(self.problem.num_steps, self.problem.movement.num_steps)

    def test_delta_time_from_movement(self):
        """Test that delta_time is taken from the movement."""
        self.assertEqual(self.problem.delta_time, self.problem.movement.delta_time)

    def test_steady_problems_initialization(self):
        """Test that steady_problems is initialized with one SteadyProblem."""
        self.assertIsInstance(self.problem.steady_problems, tuple)
        self.assertEqual(len(self.problem.steady_problems), 1)
        self.assertIsInstance(
            self.problem.steady_problems[0],
            ps.problems.SteadyProblem,
        )

    def test_initialization_of_load_lists(self):
        """Test that load lists are initialized as empty."""
        self.assertIsInstance(self.problem.forces_W, list)
        self.assertEqual(len(self.problem.forces_W), 0)

        self.assertIsInstance(self.problem.forceCoefficients_W, list)
        self.assertEqual(len(self.problem.forceCoefficients_W), 0)

        self.assertIsInstance(self.problem.moments_W_Cg, list)
        self.assertEqual(len(self.problem.moments_W_Cg), 0)

        self.assertIsInstance(self.problem.momentCoefficients_W_Cg, list)
        self.assertEqual(len(self.problem.momentCoefficients_W_Cg), 0)

    def test_I_BP1_CgP1_attribute(self):
        """Test that I_BP1_CgP1 is stored correctly."""
        self.assertIsInstance(self.problem.I_BP1_CgP1, np.ndarray)
        self.assertEqual(self.problem.I_BP1_CgP1.shape, (3, 3))
        np.testing.assert_array_equal(self.problem.I_BP1_CgP1, np.diag([1.0, 1.0, 1.0]))

    def test_mujoco_model_attribute(self):
        """Test that the MuJoCoModel is constructed during initialization."""
        # noinspection PyProtectedMember
        from pterasoftware import _mujoco_model

        self.assertIsInstance(self.problem.mujoco_model, _mujoco_model.MuJoCoModel)


class TestFreeFlightUnsteadyProblemImmutability(unittest.TestCase):
    """Tests for FreeFlightUnsteadyProblem attribute immutability."""

    def setUp(self):
        """Set up a fresh FreeFlightUnsteadyProblem for each immutability test."""
        self.problem = (
            problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()
        )

    def test_immutable_movement_property(self):
        """Test that the movement property is read only."""
        with self.assertRaises(AttributeError):
            self.problem.movement = None

    def test_immutable_I_BP1_CgP1_property(self):
        """Test that the I_BP1_CgP1 property is read only."""
        with self.assertRaises(AttributeError):
            self.problem.I_BP1_CgP1 = np.eye(3)

    def test_I_BP1_CgP1_array_not_writeable(self):
        """Test that the I_BP1_CgP1 numpy array is not writeable."""
        with self.assertRaises(ValueError):
            self.problem.I_BP1_CgP1[0, 0] = 999.0

    def test_immutable_external_forces_fn_property(self):
        """Test that the external_forces_fn property is read only."""
        with self.assertRaises(AttributeError):
            self.problem.external_forces_fn = None

    def test_immutable_mujoco_model_property(self):
        """Test that the mujoco_model property is read only."""
        with self.assertRaises(AttributeError):
            self.problem.mujoco_model = None

    def test_immutable_num_steps_property(self):
        """Test that the num_steps property is read only."""
        with self.assertRaises(AttributeError):
            self.problem.num_steps = 100

    def test_immutable_delta_time_property(self):
        """Test that the delta_time property is read only."""
        with self.assertRaises(AttributeError):
            self.problem.delta_time = 0.5

    def test_mutable_load_lists(self):
        """Test that load lists remain mutable for solver population."""
        self.problem.forces_W.append(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(len(self.problem.forces_W), 1)


if __name__ == "__main__":
    unittest.main()
