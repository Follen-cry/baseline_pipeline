"""Unit tests for scorers/multi_object_placement.py. Fixtures are synthetic, so
the tests run without the eval dataset."""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scorers'))
import multi_object_placement as M  # noqa: E402

W = 512
BG = (255, 255, 255)
BLUE = (255, 100, 100)      # BGR
GREEN = (100, 255, 100)
ORANGE = (100, 165, 255)
# (colour, shape, start position, star/target position)
SCENE = [(BLUE, 'square', (110, 400), (150, 140)),
         (GREEN, 'circle', (400, 110), (360, 300)),
         (ORANGE, 'triangle', (110, 110), (380, 430))]
SIDE = 56


def _shape(img, kind, colour, c, side=SIDE):
    x, y = c
    h = side // 2
    if kind == 'square':
        cv2.rectangle(img, (x - h, y - h), (x + h, y + h), colour, -1)
    elif kind == 'circle':
        cv2.circle(img, (x, y), h, colour, -1)
    else:
        pts = np.array([[x, y - h], [x - h, y + h], [x + h, y + h]], np.int32)
        cv2.fillPoly(img, [pts], colour)


def _star(img, colour, c, r=8):
    x, y = c
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r // 2
        a = np.pi / 2 + i * np.pi / 5
        pts.append([int(x + rad * np.cos(a)), int(y - rad * np.sin(a))])
    cv2.fillPoly(img, [np.array(pts, np.int32)], colour)


def render(placements, stars=(), bg=BG, sizes=None):
    """placements: list of (colour, kind, centre)."""
    img = np.full((W, W, 3), bg, np.uint8)
    for c in stars:
        _star(img, c[0], c[1])
    for k, (colour, kind, centre) in enumerate(placements):
        _shape(img, kind, colour, centre, (sizes or {}).get(k, SIDE))
    return img


def input_frame():
    return render([(c, k, s) for c, k, s, _ in SCENE],
                  stars=[(c, t) for c, _, _, t in SCENE])


def gt_frame():
    """Solved: every shape sits on its own star, covering it."""
    return render([(c, k, t) for c, k, _, t in SCENE])


class MultiObjectPlacementScorerTest(unittest.TestCase):
    def setUp(self):
        self.inp, self.gt = input_frame(), gt_frame()

    def s(self, cand):
        return M.score(self.inp, cand, self.gt)

    def ov(self, cand):
        return M.overall(self.s(cand))

    # ---- blank -------------------------------------------------------------
    def test_blank_scores_zero(self):
        sub = self.s(np.full((W, W, 3), BG, np.uint8))
        self.assertLess(M.overall(sub), 0.10, sub)
        self.assertEqual(sub['color_matching'], 0.0)

    # ---- perfect -----------------------------------------------------------
    def test_perfect_match(self):
        sub = self.s(gt_frame())
        for k, v in sub.items():
            self.assertGreater(v, 0.95, f'{k} penalised an exact match')

    def test_nothing_moved_scores_low(self):
        """Every object still at its start position: right elements, wrong place."""
        cand = render([(c, k, s) for c, k, s, _ in SCENE])
        sub = self.s(cand)
        self.assertEqual(sub['color_matching'], 0.0)
        self.assertLess(sub['alignment'], 0.10)
        self.assertGreater(sub['fidelity'], 0.8, 'shapes themselves are intact')

    # ---- one object missing ------------------------------------------------
    def test_one_object_missing(self):
        cand = render([(c, k, t) for c, k, _, t in SCENE[:2]])
        sub = self.s(cand)
        self.assertAlmostEqual(sub['color_matching'], 2 / 3, delta=0.05)
        self.assertLess(sub['alignment'], 0.75)
        self.assertGreater(sub['alignment'], 0.55)

    # ---- colour swapped ----------------------------------------------------
    def test_colour_swapped_gets_no_positional_credit(self):
        """Shapes in exactly the right places but two colours exchanged."""
        cand = render([(GREEN, 'square', SCENE[0][3]),
                       (BLUE, 'circle', SCENE[1][3]),
                       (ORANGE, 'triangle', SCENE[2][3])])
        sub = self.s(cand)
        self.assertLess(sub['color_matching'], 0.45,
                        'a wrong-coloured object must not earn positional credit')

    def test_substitution_is_charged_once(self):
        """One object replaced by a duplicate of another colour is ONE error."""
        cand = render([(GREEN, 'circle', SCENE[0][3]),
                       (GREEN, 'circle', SCENE[1][3]),
                       (ORANGE, 'triangle', SCENE[2][3])])
        self.assertAlmostEqual(self.s(cand)['color_matching'], 2 / 3, delta=0.05)

    def test_surplus_object_is_penalised(self):
        cand = render([(c, k, t) for c, k, _, t in SCENE] +
                      [(BLUE, 'square', (256, 480))])
        self.assertLess(self.s(cand)['color_matching'], 0.80)

    # ---- shape fidelity ----------------------------------------------------
    def test_shape_substitution_hits_fidelity_only(self):
        """Circle rendered as a square, in the right place with the right colour."""
        cand = render([(BLUE, 'square', SCENE[0][3]),
                       (GREEN, 'square', SCENE[1][3]),      # was a circle
                       (ORANGE, 'triangle', SCENE[2][3])])
        sub = self.s(cand)
        self.assertGreater(sub['color_matching'], 0.95)
        self.assertLess(sub['fidelity'], 0.85)

    def test_size_change_hits_fidelity(self):
        cand = render([(c, k, t) for c, k, _, t in SCENE], sizes={0: 24})
        self.assertLess(self.s(cand)['fidelity'], 0.90)

    # ---- alignment ---------------------------------------------------------
    def test_small_offset_tolerated(self):
        cand = render([(c, k, (t[0] + 6, t[1] + 6)) for c, k, _, t in SCENE])
        sub = self.s(cand)
        self.assertGreater(sub['color_matching'], 0.95)
        self.assertGreater(sub['alignment'], 0.80)

    def test_off_by_one_cell_shift_penalised(self):
        """A whole-scene shift of ~one object width: on no star any more."""
        cand = render([(c, k, (t[0] + 60, t[1])) for c, k, _, t in SCENE])
        sub = self.s(cand)
        self.assertLess(sub['color_matching'], 0.35)
        self.assertLess(sub['alignment'], 0.35)

    def test_alignment_is_monotonic_in_offset(self):
        seq = [self.s(render([(c, k, (t[0] + d, t[1])) for c, k, _, t in SCENE]))['alignment']
               for d in (0, 8, 16, 24, 40, 60)]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(seq, seq[1:])), seq)

    # ---- stars -------------------------------------------------------------
    def test_covered_stars_score_full_invariance(self):
        """GT shows no markers; a candidate showing none must not be penalised."""
        self.assertEqual(self.s(gt_frame())['star_invariance'], 1.0)

    def test_unmoved_star_left_visible_is_not_penalised(self):
        cand = render([(c, k, t) for c, k, _, t in SCENE[:2]],
                      stars=[(SCENE[2][0], SCENE[2][3])])
        self.assertGreater(self.s(cand)['star_invariance'], 0.95)

    def test_star_drawn_somewhere_new_is_penalised(self):
        cand = render([(c, k, t) for c, k, _, t in SCENE],
                      stars=[(BLUE, (256, 480))])
        self.assertLess(self.s(cand)['star_invariance'], 0.5)

    def test_darker_rendering_still_matches_colour(self):
        """Generated frames shift lightness; hue identity is what matters.

        Scaled by 0.85, which reproduces the magnitude actually measured in the
        data: the unmoved marker in 00085/s2 differs from its input by dE 21.4
        under the L-weighted metric. (Uniform BGR scaling also reduces chroma,
        so a harsher factor would be testing a *colour* change, not a lightness
        one -- see test_clearly_different_colour_is_rejected for that side.)
        """
        dim = [(tuple(int(x * 0.85) for x in c), k, t) for c, k, _, t in SCENE]
        self.assertGreater(self.s(render(dim))['color_matching'], 0.95)

    def test_clearly_different_colour_is_rejected(self):
        """Down-weighting L must not make distinct palette colours interchangeable."""
        swapped = [(GREEN if c == BLUE else c, k, t) for c, k, _, t in SCENE]
        self.assertLess(self.s(render(swapped))['color_matching'], 0.80)

    def test_debug_writes_visualisations(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sub, dbg = M.debug(self.inp, gt_frame(), self.gt, d, 't')
            for p in dbg['debug_images'].values():
                self.assertTrue(os.path.exists(p), p)
            self.assertIn('per_object', dbg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
