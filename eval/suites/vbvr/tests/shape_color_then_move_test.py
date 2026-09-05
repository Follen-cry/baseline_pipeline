"""Unit tests for scorers/shape_color_then_move.py. Fixtures are synthetic."""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scorers'))
import shape_color_then_move as S  # noqa: E402

W = 512
BG = (255, 255, 255)
COL_A = (60, 40, 150)      # A/D colour (BGR)
COL_B = (40, 170, 210)     # B/C/E/F colour after the recolour step
COLS = (96, 240, 384)
ROW_TOP, ROW_BOT = 170, 340
MOVE_DY = -50              # the B->C / E->F displacement
SIDE = 78


def _glyph(img, kind, colour, c, side=SIDE):
    x, y = c
    h = side // 2
    if kind == 'cross':
        t = side // 4
        cv2.rectangle(img, (x - t, y - h), (x + t, y + h), colour, -1)
        cv2.rectangle(img, (x - h, y - t), (x + h, y + t), colour, -1)
    elif kind == 'quad':
        cv2.rectangle(img, (x - h, y - h), (x + h, y + h), colour, -1)
    else:
        cv2.circle(img, (x, y), h, colour, -1)


def render(cells, bg=BG):
    """cells: dict slot -> (kind, colour, (x, y)) or None."""
    img = np.full((W, W, 3), bg, np.uint8)
    for name, spec in cells.items():
        if spec is None:
            continue
        kind, colour, c = spec
        _glyph(img, kind, colour, c)
    return img


def solved(kind_top='cross', kind_bot='quad'):
    return {
        'A': (kind_top, COL_A, (COLS[0], ROW_TOP)),
        'B': (kind_top, COL_B, (COLS[1], ROW_TOP)),
        'C': (kind_top, COL_B, (COLS[2], ROW_TOP + MOVE_DY)),
        'D': (kind_bot, COL_A, (COLS[0], ROW_BOT)),
        'E': (kind_bot, COL_B, (COLS[1], ROW_BOT)),
        'F': (kind_bot, COL_B, (COLS[2], ROW_BOT + MOVE_DY)),
    }


def input_frame():
    c = solved()
    c['E'] = None
    c['F'] = None
    return render(c)


class ShapeColorThenMoveScorerTest(unittest.TestCase):
    def setUp(self):
        self.inp = input_frame()
        self.gt = render(solved())

    def s(self, cand):
        return S.score(self.inp, cand, self.gt)

    def ov(self, cand):
        return S.overall(self.s(cand))

    # ---- blank -------------------------------------------------------------
    def test_blank_scores_zero(self):
        sub = self.s(np.full((W, W, 3), BG, np.uint8))
        self.assertLess(S.overall(sub), 0.10, sub)

    def test_input_echoed_unchanged_gets_no_completion_credit(self):
        """Copying the input back: top row intact, E and F never drawn."""
        sub = self.s(input_frame())
        self.assertGreater(sub['first_row_preservation'], 0.95)
        self.assertLess(sub['second_row_completion'], 0.40)
        self.assertEqual(sub['color_accuracy'], 0.0)

    # ---- perfect -----------------------------------------------------------
    def test_perfect_match(self):
        sub = self.s(render(solved()))
        for k, v in sub.items():
            self.assertGreater(v, 0.95, f'{k} penalised an exact match')

    # ---- single cell missing -----------------------------------------------
    def test_one_top_cell_missing(self):
        c = solved(); c['C'] = None
        sub = self.s(render(c))
        self.assertAlmostEqual(sub['first_row_preservation'], 2 / 3, delta=0.05)
        self.assertGreater(sub['second_row_completion'], 0.90,
                           'an intact bottom row must not be docked twice')

    def test_destroyed_top_row_invalidates_the_analogy(self):
        c = solved()
        for n in 'ABC':
            c[n] = None
        sub = self.s(render(c))
        self.assertLess(sub['first_row_preservation'], 0.1)
        self.assertLess(sub['second_row_completion'], 0.1,
                        'the bottom row is only meaningful against an intact example')

    def test_one_bottom_cell_missing(self):
        c = solved(); c['F'] = None
        sub = self.s(render(c))
        self.assertAlmostEqual(sub['second_row_completion'], 2 / 3, delta=0.06)
        self.assertAlmostEqual(sub['color_accuracy'], 0.5, delta=0.05)

    # ---- colour ------------------------------------------------------------
    def test_wrong_colour_in_E_and_F(self):
        c = solved()
        c['E'] = ('quad', COL_A, (COLS[1], ROW_BOT))
        c['F'] = ('quad', COL_A, (COLS[2], ROW_BOT + MOVE_DY))
        sub = self.s(render(c))
        self.assertEqual(sub['color_accuracy'], 0.0)
        self.assertLess(sub['second_row_completion'], 0.40)

    def test_sequence_shifted_by_one_cell(self):
        """The failure seen in 00000/base: D takes B's colour and E takes D's."""
        c = solved()
        c['D'] = ('quad', (200, 150, 40), (COLS[0], ROW_BOT))
        c['E'] = ('quad', COL_A, (COLS[1], ROW_BOT))
        sub = self.s(render(c))
        self.assertLess(sub['second_row_completion'], 0.45)
        self.assertAlmostEqual(sub['color_accuracy'], 0.5, delta=0.05)

    # ---- the move ----------------------------------------------------------
    def test_move_not_applied_to_F(self):
        """F drawn in line with E instead of displaced: the 'move' step failed."""
        c = solved(); c['F'] = ('quad', COL_B, (COLS[2], ROW_BOT))
        sub = self.s(render(c))
        self.assertLess(sub['second_row_completion'], 0.85)

    def test_small_jitter_tolerated(self):
        c = {k: (v[0], v[1], (v[2][0] + 5, v[2][1] + 4)) for k, v in solved().items()}
        self.assertGreater(self.ov(render(c)), 0.90)

    # ---- glyph identity ----------------------------------------------------
    def test_wrong_glyph_in_bottom_row(self):
        """A glyph substitution is penalised on the cell it happens in.

        Asserted per cell rather than on the 3-cell mean: one bad cell out of
        three is diluted to ~0.9 by the mean, which says nothing about whether
        the shape test works. A circle inscribed in a square overlaps it by
        pi/4 = 0.785 -- between the measured "correct regeneration" (p5 0.85)
        and "different glyph" (p50 0.448) populations -- so it should score as
        partial, not as either extreme.
        """
        import tempfile
        c = solved()
        c['E'] = ('circle', COL_B, (COLS[1], ROW_BOT))
        with tempfile.TemporaryDirectory() as d:
            _, dbg = S.debug(self.inp, render(c), self.gt, d, 't')
        e = dbg['cells']['E']
        self.assertLess(e['score'], 0.85, 'inscribed circle scored as a glyph match')
        self.assertGreater(e['score'], 0.35, 'inscribed circle is not a total mismatch')
        self.assertGreater(dbg['cells']['D']['score'], 0.95, 'other cells unaffected')

    def test_clearly_different_glyph_in_bottom_row(self):
        """A cross where a filled square belongs.

        Asserted on the cell, not the row mean. `second_row_completion` is a
        mean over D/E/F by design -- one destroyed cell out of three costs about
        0.14, which is the behaviour verified against the real samples (00019:
        E miscoloured -> 0.64) -- so a row-level assertion would be testing the
        averaging, not the glyph comparison.
        """
        import tempfile
        c = solved()
        c['E'] = ('cross', COL_B, (COLS[1], ROW_BOT))
        with tempfile.TemporaryDirectory() as d:
            _, dbg = S.debug(self.inp, render(c), self.gt, d, 't')
        self.assertLess(dbg['cells']['E']['score'], 0.65)

    def test_size_change_penalised(self):
        img = render({k: v for k, v in solved().items() if k != 'F'})
        _glyph(img, 'quad', COL_B, (COLS[2], ROW_BOT + MOVE_DY), side=40)
        self.assertLess(self.s(img)['second_row_completion'], 0.85)

    # ---- global -----------------------------------------------------------
    def test_repainted_background_is_floored(self):
        self.assertLess(self.ov(render(solved(), bg=(0, 0, 0))), 0.20)

    def test_shape_count(self):
        c = solved(); c['E'] = None; c['F'] = None
        self.assertLess(self.s(render(c))['shape_count'], 0.75)

    def test_debug_writes_visualisations(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sub, dbg = S.debug(self.inp, render(solved()), self.gt, d, 't')
            for p in dbg['debug_images'].values():
                self.assertTrue(os.path.exists(p), p)
            self.assertIn('cells', dbg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
