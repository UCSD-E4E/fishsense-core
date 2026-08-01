"""Pure-logic tests for the classical slate estimator's geometry helpers.

Ported from ``tests/test_baseline.py`` in
``UCSD-E4E/2026-07-31_slate_training`` alongside the estimator itself.
"""

import cv2
import numpy as np
import pytest

from fishsense_core.slate.estimator import (
    homography_from_quad,
    order_quad,
    rotate_quad,
    segment_board,
    template_corners,
)


class TestOrderQuad:
    def test_orders_to_tl_tr_br_bl(self):
        scrambled = [[100, 100], [10, 90], [90, 10], [0, 0]]
        assert np.allclose(
            order_quad(scrambled), [[0, 0], [90, 10], [100, 100], [10, 90]]
        )

    def test_is_invariant_to_input_order(self):
        pts = [[0, 0], [90, 10], [100, 100], [10, 90]]
        first = order_quad(pts)
        for roll in range(4):
            assert np.allclose(order_quad(np.roll(pts, roll, axis=0)), first)

    def test_handles_a_rotated_board(self):
        # A diamond -- still must produce a consistent, non-degenerate order.
        ordered = order_quad([[50, 0], [100, 50], [50, 100], [0, 50]])
        assert ordered.shape == (4, 2)
        assert len(np.unique(ordered, axis=0)) == 4


class TestRotateQuad:
    def test_rotation_by_zero_is_identity(self):
        quad = order_quad([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert np.allclose(rotate_quad(quad, 0), quad)

    def test_four_rotations_return_to_start(self):
        quad = order_quad([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert np.allclose(rotate_quad(quad, 4), quad)

    def test_rotation_cycles_corners(self):
        quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        assert np.allclose(rotate_quad(quad, 1)[0], [10, 0])


class TestHomographyFromQuad:
    def test_identity_quad_gives_identity_mapping(self):
        quad = template_corners(100, 50)
        homography = homography_from_quad((100, 50), quad)
        mapped = cv2.perspectiveTransform(
            np.array([[[25.0, 10.0]]], dtype=np.float32), homography
        )
        assert np.allclose(mapped.reshape(2), [25.0, 10.0], atol=1e-6)

    def test_maps_template_corners_onto_the_quad(self):
        quad = np.array([[10, 20], [110, 25], [105, 75], [15, 70]], dtype=float)
        homography = homography_from_quad((100, 50), quad)
        corners = template_corners(100, 50).reshape(-1, 1, 2).astype(np.float32)
        mapped = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
        assert np.allclose(mapped, quad, atol=1e-6)

    def test_a_scaled_board_scales_interior_points(self):
        # Template 100x50 onto a 200x100 quad -> interior points double.
        quad = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=float)
        homography = homography_from_quad((100, 50), quad)
        mapped = cv2.perspectiveTransform(
            np.array([[[50.0, 25.0]]], dtype=np.float32), homography
        )
        assert np.allclose(mapped.reshape(2), [100.0, 50.0], atol=1e-6)


class TestSegmentBoard:
    def _scene(self, corners):
        """Dark background with one bright quad, like a board against water."""
        img = np.full((400, 400, 3), 30, np.uint8)
        cv2.fillPoly(img, [np.asarray(corners, np.int32)], (235, 235, 235))
        return img

    def test_finds_an_axis_aligned_board(self):
        corners = [[100, 80], [300, 80], [300, 220], [100, 220]]
        quad = segment_board(self._scene(corners))
        assert quad is not None
        assert np.allclose(quad, order_quad(corners), atol=6)

    def test_finds_a_rotated_board(self):
        corners = [[120, 90], [310, 130], [290, 250], [100, 210]]
        quad = segment_board(self._scene(corners))
        assert quad is not None
        assert np.allclose(quad, order_quad(corners), atol=8)

    def test_ignores_specks_below_the_area_floor(self):
        # Backscatter and glints are small and bright; they must not win.
        img = np.full((400, 400, 3), 30, np.uint8)
        cv2.rectangle(img, (10, 10), (12, 12), (235, 235, 235), -1)
        assert segment_board(img) is None

    def test_rejects_a_region_covering_the_whole_frame(self):
        # A white-out (or a bad threshold) must not be reported as a board.
        assert segment_board(np.full((400, 400, 3), 240, np.uint8)) is None

    def test_returns_none_on_a_featureless_frame(self):
        assert segment_board(np.full((400, 400, 3), 30, np.uint8)) is None

    def test_dark_pattern_inside_the_board_does_not_split_it(self):
        # The printed chevron must not fragment the board into pieces.
        corners = [[100, 80], [300, 80], [300, 220], [100, 220]]
        img = self._scene(corners)
        cv2.fillPoly(
            img,
            [np.array([[150, 100], [200, 190], [250, 100]], np.int32)],
            (25, 25, 25),
        )
        quad = segment_board(img)
        assert quad is not None
        assert np.allclose(quad, order_quad(corners), atol=8)


class TestRescaleHomography:
    """Lifting a reduced-resolution fit back to full size."""

    def test_unit_scales_are_a_no_op(self):
        from fishsense_core.slate.estimator import rescale_homography
        h = np.array([[1.0, 0.2, 5.0], [0.1, 1.0, 3.0], [0.0, 0.0, 1.0]])
        assert np.allclose(rescale_homography(h, 1.0, 1.0), h)

    def test_round_trips_a_point_through_the_scaled_fit(self):
        from fishsense_core.slate.estimator import rescale_homography
        # Full-res truth: template point -> image point.
        truth = np.array([[2.0, 0.0, 30.0], [0.0, 2.0, 40.0], [0.0, 0.0, 1.0]])
        t_scale, i_scale = 0.25, 0.5
        # The equivalent small-scale warp, per the derivation in the docstring.
        small = (
            np.diag([i_scale, i_scale, 1.0]) @ truth
            @ np.linalg.inv(np.diag([t_scale, t_scale, 1.0]))
        )
        lifted = rescale_homography(small, t_scale, i_scale)
        assert np.allclose(lifted, truth, atol=1e-9)

    def test_is_normalised_to_unit_bottom_right(self):
        from fishsense_core.slate.estimator import rescale_homography
        h = np.array([[3.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]])
        assert rescale_homography(h, 0.5, 0.25)[2, 2] == pytest.approx(1.0)


class TestRefineAtFullRes:
    """The crop-and-refine pass that targets plane offset (scale) accuracy."""

    def _template(self):
        t = np.full((200, 300), 255, np.uint8)
        cv2.fillPoly(t, [np.array([[60, 40], [240, 40], [150, 160]], np.int32)], 0)
        return t

    def test_recovers_a_perturbed_homography(self):
        from fishsense_core.slate.estimator import refine_at_full_res
        template = self._template()
        scene = np.full((900, 1200), 235, np.uint8)
        truth = np.array([[0.8, 0.0, 400.0], [0.0, 0.8, 300.0], [0.0, 0.0, 1.0]])
        cv2.warpPerspective(template, truth, (1200, 900), dst=scene,
                            borderMode=cv2.BORDER_TRANSPARENT)
        # Start a few pixels off and check refinement moves toward truth.
        seed = truth @ np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
        refined, score = refine_at_full_res(scene, template, seed)
        probe = np.array([[[150.0, 100.0]]], np.float32)
        target = cv2.perspectiveTransform(probe, truth).reshape(2)
        before = np.linalg.norm(cv2.perspectiveTransform(probe, seed).reshape(2) - target)
        after = np.linalg.norm(cv2.perspectiveTransform(probe, refined).reshape(2) - target)
        assert score > 0.0
        assert after < before

    def test_degenerate_homography_is_returned_unchanged(self):
        from fishsense_core.slate.estimator import refine_at_full_res
        template = self._template()
        scene = np.full((400, 400), 200, np.uint8)
        # Collapses the board to a sub-pixel speck -> nothing to refine.
        tiny = np.array([[1e-4, 0, 10.0], [0, 1e-4, 10.0], [0, 0, 1.0]])
        refined, score = refine_at_full_res(scene, template, tiny)
        assert score == 0.0
        assert np.allclose(refined, tiny)

    def test_board_projected_outside_the_frame_is_handled(self):
        from fishsense_core.slate.estimator import refine_at_full_res
        template = self._template()
        scene = np.full((400, 400), 200, np.uint8)
        outside = np.array([[1.0, 0, 5000.0], [0, 1.0, 5000.0], [0, 0, 1.0]])
        refined, score = refine_at_full_res(scene, template, outside)
        assert score == 0.0
        assert np.allclose(refined, outside)


class TestTemplatePatternQuad:
    """The patch the search actually matches against."""

    def _render(self):
        # Page with a border (the trap) plus a small central pattern.
        t = np.full((300, 400), 255, np.uint8)
        cv2.rectangle(t, (2, 2), (397, 297), 0, 3)          # page border
        cv2.fillPoly(t, [np.array([[160, 120], [240, 120], [200, 180]],
                                  np.int32)], 0)            # the pattern
        return t

    def test_dark_pixel_fallback_spans_the_whole_page(self):
        # Documents the trap: the border makes "all dark pixels" the page.
        from fishsense_core.slate.estimator import template_pattern_quad
        quad = template_pattern_quad(self._render())
        assert np.ptp(quad[:, 0]) > 380
        assert np.ptp(quad[:, 1]) > 280

    def test_reference_points_give_a_tight_pattern_box(self):
        from fishsense_core.slate.estimator import template_pattern_quad
        pts = [(160.0, 120.0), (240.0, 120.0), (200.0, 180.0)]
        quad = template_pattern_quad(self._render(), pts)
        assert np.ptp(quad[:, 0]) < 120     # ~80px + padding, not 400
        assert np.ptp(quad[:, 1]) < 100

    def test_padding_widens_beyond_the_points(self):
        from fishsense_core.slate.estimator import template_pattern_quad
        pts = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
        tight = template_pattern_quad(self._render(), pts, pad_frac=0.0)
        padded = template_pattern_quad(self._render(), pts, pad_frac=0.2)
        assert np.ptp(padded[:, 0]) > np.ptp(tight[:, 0])

    def test_scale_converts_point_coordinates(self):
        from fishsense_core.slate.estimator import template_pattern_quad
        pts = [(200.0, 200.0), (400.0, 200.0), (400.0, 400.0), (200.0, 400.0)]
        quad = template_pattern_quad(self._render(), pts, scale=0.5, pad_frac=0.0)
        assert np.ptp(quad[:, 0]) == pytest.approx(100.0, abs=1.0)

    def test_stays_inside_the_render(self):
        # Generous padding must clamp to the render, not run off the edge.
        from fishsense_core.slate.estimator import template_pattern_quad
        pts = [(5.0, 5.0), (395.0, 5.0), (395.0, 295.0), (5.0, 295.0)]
        quad = template_pattern_quad(self._render(), pts, pad_frac=0.5)
        assert quad[:, 0].min() >= 0 and quad[:, 0].max() <= 400
        assert quad[:, 1].min() >= 0 and quad[:, 1].max() <= 300

    def test_too_few_points_falls_back(self):
        from fishsense_core.slate.estimator import template_pattern_quad
        quad = template_pattern_quad(self._render(), [(10.0, 10.0)])
        assert np.ptp(quad[:, 0]) > 380


class TestMaskCandidates:
    """Learned-mask candidates: additive, never a replacement."""

    def _mask(self, boxes, shape=(96, 128)):
        m = np.zeros(shape, np.uint8)
        for x0, y0, x1, y1 in boxes:
            m[y0:y1, x0:x1] = 255
        return m

    def test_returns_a_quad_around_the_blob(self):
        from fishsense_core.slate.estimator import mask_candidates
        quads = mask_candidates(self._mask([(40, 30, 70, 60)]), (96, 128))
        assert len(quads) == 1
        assert quads[0][:, 0].min() < 40 and quads[0][:, 0].max() > 70

    def test_scales_the_mask_to_the_target_shape(self):
        from fishsense_core.slate.estimator import mask_candidates
        quads = mask_candidates(self._mask([(40, 30, 70, 60)]), (960, 1280))
        # 10x larger frame -> quad scales with it.
        assert quads[0][:, 0].max() > 700

    def test_orders_components_largest_first(self):
        from fishsense_core.slate.estimator import mask_candidates
        quads = mask_candidates(
            self._mask([(2, 2, 10, 10), (40, 30, 90, 80)]), (96, 128))
        first = (quads[0][:, 0].max() - quads[0][:, 0].min())
        assert first > 40

    def test_empty_mask_yields_no_candidates(self):
        from fishsense_core.slate.estimator import mask_candidates
        assert mask_candidates(np.zeros((96, 128), np.uint8), (96, 128)) == []

    def test_accepts_float_probabilities(self):
        from fishsense_core.slate.estimator import mask_candidates
        probs = self._mask([(40, 30, 70, 60)]).astype(np.float32) / 255.0
        assert len(mask_candidates(probs, (96, 128))) == 1

    def test_respects_the_limit(self):
        from fishsense_core.slate.estimator import mask_candidates
        boxes = [(2 + 20 * i, 2, 16 + 20 * i, 16) for i in range(6)]
        assert len(mask_candidates(self._mask(boxes), (96, 128), limit=3)) == 3

    def test_tiny_specks_are_dropped(self):
        from fishsense_core.slate.estimator import mask_candidates
        assert mask_candidates(self._mask([(5, 5, 7, 7)]), (96, 128)) == []


class TestShrinkToPattern:
    """Board-quad -> pattern-quad conversion for learned mask candidates."""

    def test_unit_ratio_is_identity(self):
        from fishsense_core.slate.estimator import shrink_to_pattern
        q = np.array([[0., 0.], [100., 0.], [100., 60.], [0., 60.]])
        assert np.allclose(shrink_to_pattern(q, 1.0, 1.0), q)

    def test_shrinks_about_the_centre(self):
        from fishsense_core.slate.estimator import shrink_to_pattern
        q = np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]])
        out = shrink_to_pattern(q, 0.5, 0.5)
        assert np.allclose(out.mean(axis=0), q.mean(axis=0))
        assert np.ptp(out[:, 0]) == pytest.approx(50.0)

    def test_axes_shrink_independently(self):
        from fishsense_core.slate.estimator import shrink_to_pattern
        q = np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]])
        out = shrink_to_pattern(q, 0.8, 0.5)
        assert np.ptp(out[:, 0]) == pytest.approx(80.0)
        assert np.ptp(out[:, 1]) == pytest.approx(50.0)


class TestEstimatePlaneEndToEnd:
    """The full estimate_plane pipeline (segment/search -> ECC -> solvePnP ->
    plane). The source suite covers only the pure helpers; this exercises the
    OpenCV pipeline and the plane_from_pose wiring into fishsense_core.plane."""

    def _template_and_points(self):
        tpl = np.full((300, 400), 255, np.uint8)
        pts = [(140.0, 95.0), (275.0, 110.0), (255.0, 215.0), (130.0, 200.0)]
        cv2.fillPoly(tpl, [np.array(pts, np.int32)], 0)
        cv2.circle(tpl, (200, 150), 8, 0, -1)  # break symmetry
        return tpl, pts

    def test_recovers_a_board_plane_from_a_rendered_scene(self):
        from fishsense_core.slate import estimate_plane, BoardEstimate
        from fishsense_core.plane import Plane

        tpl, pts = self._template_and_points()
        k = np.array([[1400.0, 0, 960.0], [0, 1400.0, 720.0], [0, 0, 1.0]])
        homography = np.array([[1.6, 0.05, 700.0], [0.03, 1.6, 480.0], [0, 0, 1.0]])
        scene = np.full((1440, 1920), 40, np.uint8)
        cv2.warpPerspective(tpl, homography, (1920, 1440), dst=scene,
                            borderMode=cv2.BORDER_TRANSPARENT)
        bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

        est = estimate_plane(bgr, tpl, pts, dpi=100.0, camera_matrix=k)
        assert isinstance(est, BoardEstimate)
        assert isinstance(est.plane, Plane)
        assert np.isclose(np.linalg.norm(est.plane.normal), 1.0)  # plane wiring
        assert est.image_points.shape == (len(pts), 2)
        assert est.plane.distance > 0  # board is in front of the camera

    def test_featureless_frame_returns_none(self):
        from fishsense_core.slate import estimate_plane

        tpl, pts = self._template_and_points()
        k = np.array([[1400.0, 0, 960.0], [0, 1400.0, 720.0], [0, 0, 1.0]])
        flat = np.full((1440, 1920, 3), 40, np.uint8)  # no contrast, no board
        assert estimate_plane(flat, tpl, pts, dpi=100.0, camera_matrix=k) is None
