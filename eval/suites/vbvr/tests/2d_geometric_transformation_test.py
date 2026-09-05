"""Unit tests for scorers/2d_geometric_transformation.py. Fixtures are synthetic."""
import importlib
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scorers'))
G = importlib.import_module('2d_geometric_transformation')

W = 512
BG = (240, 240, 240)
SHAPE_COL = (135, 160, 132)          # deliberately muted, like the real palette
OUTLINE_COL = (150, 150, 150)
START = (170, 300)
TARGET = (300, 230)                  # required move ~147 px... trimmed below
TARGET = (255, 255)                  # ~113 px
START_ANGLE, TARGET_ANGLE = 0.0, 35.0
POLY = np.array([[-28, -18], [26, -22], [30, 14], [-6, 22], [-24, 10]], np.float32)


def _poly_at(centre, angle_deg, scale=1.0):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    return (POLY * scale @ R.T + np.array(centre, np.float32)).astype(np.int32)


def render(shape=None, outline=None, pivot=True, bg=BG, colour=SHAPE_COL):
    """shape / outline: (centre, angle[, scale]) or None."""
    img = np.full((W, W, 3), bg, np.uint8)
    if outline is not None:
        cv2.polylines(img, [_poly_at(*outline)], True, OUTLINE_COL, 2)
    if pivot:
        cv2.circle(img, START, 5, (255, 255, 255), -1)
        cv2.circle(img, START, 5, (90, 90, 90), 1)
    if shape is not None:
        cv2.fillPoly(img, [_poly_at(*shape)], colour)
    return img


def input_frame():
    return render(shape=(START, START_ANGLE), outline=(TARGET, TARGET_ANGLE))


def gt_frame():
    return render(shape=(TARGET, TARGET_ANGLE), outline=None)


class GeometricTransformationScorerTest(unittest.TestCase):
    def setUp(self):
        self.inp, self.gt = input_frame(), gt_frame()

    def s(self, cand):
        return G.score(self.inp, cand, self.gt)

    def ov(self, cand):
        return G.overall(self.s(cand))

    def dbg(self, cand):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            return G.debug(self.inp, cand, self.gt, d, 't')

    # ---- blank -------------------------------------------------------------
    def test_blank_scores_zero(self):
        sub = self.s(render(shape=None, outline=None, pivot=False))
        self.assertLess(G.overall(sub), 0.05, sub)

    def test_only_the_outline_left_scores_zero(self):
        """The failure in 00020/s1: the shape itself is gone."""
        sub = self.s(render(shape=None, outline=(TARGET, TARGET_ANGLE)))
        self.assertLess(G.overall(sub), 0.10)

    # ---- perfect -----------------------------------------------------------
    def test_perfect_match(self):
        sub = self.s(gt_frame())
        for k, v in sub.items():
            self.assertGreater(v, 0.90, f'{k} penalised an exact match')

    # ---- the "nothing happened" case ---------------------------------------
    def test_shape_left_at_start_gets_no_position_credit(self):
        """Right element, wrong place: position must be 0, not a floor value."""
        sub = self.s(render(shape=(START, START_ANGLE), outline=(TARGET, TARGET_ANGLE)))
        self.assertLess(sub['position_alignment'], 0.05,
                        'a shape that never moved must score 0 on position')

    def test_input_echoed_back_scores_low(self):
        self.assertLess(self.ov(input_frame()), 0.30)

    def test_no_position_credit_even_when_the_move_is_short(self):
        """A short required move must not let 'did nothing' score well.

        This is the 00010/s0 regression: an earlier version bounded the position
        ramp by shape size, so on a sample whose required move was 17 px the
        untouched shape still scored 0.34.
        """
        near = (START[0] + 22, START[1])
        gt = render(shape=(near, START_ANGLE))
        sub = G.score(render(shape=(START, START_ANGLE), outline=(near, START_ANGLE)),
                      render(shape=(START, START_ANGLE)), gt)
        self.assertLess(sub['position_alignment'], 0.05)

    # ---- rotation ----------------------------------------------------------
    def test_right_place_wrong_rotation(self):
        sub = self.s(render(shape=(TARGET, TARGET_ANGLE + 90)))
        self.assertGreater(sub['position_alignment'], 0.90)
        self.assertLess(sub['rotation_angle'], 0.05)

    def test_small_rotation_error_tolerated(self):
        self.assertGreater(self.s(render(shape=(TARGET, TARGET_ANGLE + 5)))['rotation_angle'], 0.90)

    def test_rotation_is_monotonic_in_error(self):
        seq = [self.s(render(shape=(TARGET, TARGET_ANGLE + d)))['rotation_angle']
               for d in (0, 6, 12, 20, 30, 50)]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(seq, seq[1:])), seq)

    def test_flipped_180_is_caught(self):
        self.assertLess(self.s(render(shape=(TARGET, TARGET_ANGLE + 180)))['rotation_angle'], 0.05)

    # ---- position ----------------------------------------------------------
    def test_position_is_monotonic_in_offset(self):
        seq = [self.s(render(shape=((TARGET[0] + d, TARGET[1]), TARGET_ANGLE)))['position_alignment']
               for d in (0, 8, 16, 30, 60, 110)]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(seq, seq[1:])), seq)

    def test_small_offset_tolerated(self):
        self.assertGreater(
            self.s(render(shape=((TARGET[0] + 4, TARGET[1] + 3), TARGET_ANGLE)))['position_alignment'],
            0.85)

    # ---- fidelity ----------------------------------------------------------
    def test_scaled_shape_loses_fidelity(self):
        """A shape at 0.6x scale, correctly placed.

        Position keeps most of its credit -- the object is where it belongs --
        but not all of it: at 0.6x the best-fit silhouette overlap is 0.377, so
        the identity gate treats it as only partly the same object. Fidelity
        carries the bulk of the penalty, which is where a size error belongs.
        """
        sub = self.s(render(shape=(TARGET, TARGET_ANGLE, 0.6)))
        self.assertLess(sub['shape_fidelity'], 0.60)
        self.assertGreater(sub['position_alignment'], 0.60, 'position is still broadly right')

    def test_deformed_shape_loses_fidelity(self):
        """A long thin bar where the polygon belongs: silhouette clearly wrong."""
        img = np.full((W, W, 3), BG, np.uint8)
        cv2.circle(img, START, 5, (255, 255, 255), -1)
        cv2.fillPoly(img, [np.array([[TARGET[0] - 46, TARGET[1] - 8],
                                     [TARGET[0] + 46, TARGET[1] - 8],
                                     [TARGET[0] + 46, TARGET[1] + 8],
                                     [TARGET[0] - 46, TARGET[1] + 8]], np.int32)], SHAPE_COL)
        self.assertLess(self.s(img)['shape_fidelity'], 0.70)

    def test_circle_substituted_for_polygon_is_only_partial(self):
        """A circle overlaps this polygon by 0.778 at best fit -- genuinely
        similar, so it reads as partial rather than as a clean failure. Recorded
        as a known limit of a silhouette-overlap fidelity measure."""
        img = np.full((W, W, 3), BG, np.uint8)
        cv2.circle(img, TARGET, 28, SHAPE_COL, -1)
        cv2.circle(img, START, 5, (255, 255, 255), -1)
        self.assertLess(self.s(img)['shape_fidelity'], 1.0)

    # ---- residual outline --------------------------------------------------
    def test_leftover_target_outline_is_detected(self):
        clean = self.dbg(gt_frame())[1]
        left = self.dbg(render(shape=(TARGET, TARGET_ANGLE),
                               outline=(TARGET, TARGET_ANGLE)))[1]
        self.assertLess(clean['residual']['outline_retained'], 0.15)
        self.assertGreater(left['residual']['outline_retained'], 0.30,
                           'a retained target outline must be measured')
        self.assertLess(G.overall(self.s(render(shape=(TARGET, TARGET_ANGLE),
                                                outline=(TARGET, TARGET_ANGLE)))),
                        G.overall(self.s(gt_frame())))

    # ---- colour / scene ----------------------------------------------------
    def test_wrong_colour_is_penalised(self):
        right = self.ov(gt_frame())
        wrong = self.ov(render(shape=(TARGET, TARGET_ANGLE), colour=(40, 40, 200)))
        self.assertLess(wrong, right * 0.75)

    def test_repainted_background_is_floored(self):
        self.assertLess(self.ov(render(shape=(TARGET, TARGET_ANGLE), bg=(0, 0, 0))), 0.20)

    def test_muted_palette_is_detected_at_all(self):
        """The old scorer's `hsv[:,:,1] > 50` found nothing in this palette."""
        _, d = self.dbg(gt_frame())
        self.assertTrue(d['found']['gt'] and d['found']['cand'] and d['found']['input'])

    def test_debug_writes_visualisations(self):
        _, d = self.dbg(gt_frame())
        self.assertIn('rotation', d)
        self.assertIn('position', d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
