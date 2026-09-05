"""Unit tests for scorers/rotation_puzzle.py.

Self-contained: every fixture is rendered synthetically here, so the tests run
without the eval dataset and assert the *discrimination* properties the scorer
is supposed to have (a scorer that returns 1.0 for everything, or 0.55 for
everything, fails these).
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scorers'))
import rotation_puzzle as RP  # noqa: E402

BG = (248, 250, 252)
TILE = (255, 255, 255)
PIPE = (0, 165, 255)          # orange, BGR
CELLS = RP.FALLBACK_CELLS
# the solved ring: each cell's two arms point at its inward neighbours
SOLVED = {'TL': 'ES', 'TR': 'SW', 'BL': 'EN', 'BR': 'NW'}
STROKE = 6


def render(arms_by_cell, tiles=True, pipe=PIPE, bg=BG, arm_frac=1.0, shift=(0, 0)):
    """Draw a 2x2 rotation puzzle. `arm_frac` shortens every arm toward the
    cell centre (1.0 reaches the tile edge); `shift` translates the whole grid."""
    img = np.full((512, 512, 3), bg, np.uint8)
    dx, dy = shift
    for name, (x, y, w, h) in CELLS.items():
        x, y = x + dx, y + dy
        if tiles:
            cv2.rectangle(img, (x, y), (x + w, y + h), TILE, -1)
        cx, cy = x + w // 2, y + h // 2
        for d in arms_by_cell.get(name, ''):
            if d == 'N':
                end = (cx, int(cy - arm_frac * h / 2))
            elif d == 'S':
                end = (cx, int(cy + arm_frac * h / 2))
            elif d == 'W':
                end = (int(cx - arm_frac * w / 2), cy)
            else:
                end = (int(cx + arm_frac * w / 2), cy)
            cv2.line(img, (cx, cy), end, pipe, STROKE)
    return img


def unsolved():
    return render({'TL': 'NW', 'TR': 'NE', 'BL': 'SW', 'BR': 'SE'})


class RotationPuzzleScorerTest(unittest.TestCase):
    def setUp(self):
        self.inp = unsolved()
        self.gt = render(SOLVED)

    def s(self, cand):
        return RP.score(self.inp, cand, self.gt)

    def ov(self, cand):
        return RP.overall(self.s(cand))

    # ---- edge case: blank -------------------------------------------------
    def test_blank_image_scores_near_zero(self):
        blank = np.full((512, 512, 3), BG, np.uint8)
        sub = self.s(blank)
        self.assertLess(RP.overall(sub), 0.10, sub)
        for k, v in sub.items():
            self.assertLessEqual(v, 0.25, f'{k} too generous on a blank image')

    def test_blank_tiles_no_pipes_scores_low(self):
        """Tiles present but every pipe erased: the global ceiling must floor it
        even though position_preservation legitimately sees its tiles."""
        cand = render({}, tiles=True)
        self.assertLess(self.ov(cand), 0.15)

    # ---- perfect match ----------------------------------------------------
    def test_perfect_match_scores_one(self):
        sub = self.s(render(SOLVED))
        for k, v in sub.items():
            self.assertGreater(v, 0.95, f'{k} penalised an exact match')
        self.assertGreater(RP.overall(sub), 0.95)

    # ---- single object missing -------------------------------------------
    def test_one_tile_pipe_missing(self):
        arms = dict(SOLVED)
        del arms['BR']
        sub = self.s(render(arms))
        self.assertLess(sub['rotation_accuracy'], 0.80)
        self.assertLess(sub['path_connection'], 0.75)
        self.assertGreater(RP.overall(sub), 0.40, 'three correct cells deserve partial credit')
        self.assertLess(RP.overall(sub), 0.80)

    # ---- wrong rotation ---------------------------------------------------
    def test_one_tile_rotated_wrong(self):
        arms = dict(SOLVED, BR='SE')      # opens outward instead of inward
        sub = self.s(render(arms))
        self.assertLess(sub['rotation_accuracy'], 0.80)
        self.assertLess(sub['path_connection'], 0.60,
                        'a cell opening outward breaks two junctions and dangles')

    def test_all_tiles_rotated_wrong(self):
        arms = {'TL': 'NW', 'TR': 'NE', 'BL': 'SW', 'BR': 'SE'}
        sub = self.s(render(arms))
        self.assertLess(sub['rotation_accuracy'], 0.30)
        self.assertLess(sub['path_connection'], 0.10)

    # ---- colour swapped ---------------------------------------------------
    def test_colour_swapped_is_penalised(self):
        right = self.ov(render(SOLVED))
        wrong = self.ov(render(SOLVED, pipe=(255, 0, 0)))   # blue instead of orange
        self.assertLess(wrong, right * 0.75,
                        'a recoloured but structurally correct ring must lose credit')

    def test_background_repainted_is_floored(self):
        cand = render(SOLVED, bg=(0, 0, 0), tiles=False)
        self.assertLess(self.ov(cand), 0.20, 'scene repainted on black must be floored')

    # ---- off-by-one / geometric shift -------------------------------------
    def test_small_shift_tolerated(self):
        """A 3px whole-grid jitter is rendering noise, not an error."""
        self.assertGreater(self.ov(render(SOLVED, shift=(3, 3))), 0.85)

    def test_large_shift_penalised(self):
        """Half a tile off: the arms no longer meet where GT says they meet."""
        self.assertLess(self.ov(render(SOLVED, shift=(55, 0))), 0.75)

    # ---- short arms (the connectivity question the rewrite is built on) ---
    def test_arms_that_do_not_reach_break_the_path(self):
        short = render(SOLVED, arm_frac=0.70)   # stops ~16px from the tile edge
        sub = self.s(short)
        self.assertLess(sub['path_connection'], 0.30,
                        'arms that stop short of the gutter are not connected')

    def test_arms_reaching_within_a_stroke_still_connect(self):
        near = render(SOLVED, arm_frac=0.94)    # ~3px short, within ARM_GAP_PX
        self.assertGreater(self.s(near)['path_connection'], 0.90)

    # ---- monotonicity ------------------------------------------------------
    def test_more_correct_cells_scores_higher(self):
        seq = []
        for wrong in range(5):
            arms = dict(SOLVED)
            for k in list(SOLVED)[:wrong]:
                arms[k] = {'TL': 'NW', 'TR': 'NE', 'BL': 'SW', 'BR': 'SE'}[k]
            seq.append(self.ov(render(arms)))
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(seq, seq[1:])),
                        f'score must not increase as cells get more wrong: {seq}')

    def test_debug_writes_visualisations(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sub, dbg = RP.debug(self.inp, render(SOLVED), self.gt, d, 't')
            for p in dbg['debug_images'].values():
                self.assertTrue(os.path.exists(p), p)
            self.assertIn('cand_gaps', dbg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
