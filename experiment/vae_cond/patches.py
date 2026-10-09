"""Runtime patches for the stock InternVL-U package (the package itself is not edited).

allow_multi_image_uncond: the processor's `_insert_media_placeholders` asserts "only 1 fake picture in pure text input"
for the unconditional (drop-all) CFG row, whose prompt has no <image> placeholders. With 3 input frames that assert
fires (3 frames -> 3 patch groups). That row's image data is discarded by `__call__` (only the none/text rows append
`image_data`), so we return exactly what the original returns for the 1-image case, without the assert.
"""


def allow_multi_image_uncond(cls):
    """Patch the InternVLUProcessor *class* (call before the pipeline builds/uses a processor)."""
    if getattr(cls, "_vae_cond_patched", False):
        return
    orig = cls._insert_media_placeholders

    def patched(self, text, image_pixel_values, image_num_patches, image_num_patches_indices):
        if image_pixel_values is not None and not any(self.image_token in t for t in text):
            return list(text), [image_pixel_values], 0  # what the original does for the 1-image case
        return orig(self, text, image_pixel_values, image_num_patches, image_num_patches_indices)

    cls._insert_media_placeholders = patched
    cls._vae_cond_patched = True


def fix_multi_cond_token_slice(cls):
    """Make the packed decoder path support >1 VAE condition image per sample (needed for the `all` condition).

    `InternVLUTransformer2DModel._prepare_hidden_inputs_anyres` slices the packed condition tokens with an end index of
    `start + num_post_image_token_gen_cond[cur:cur+n]`, i.e. adding a *vector* of per-image token counts; it only works
    for n == 1 (TypeError: only integer tensors of a single element can be converted to an index, at n == 3). The
    intended value is the sum of those counts, which is also what the cursor update needs. We rewrite the function's
    source with `.sum()` added at those two places (and assert exactly two were found), leaving the package unedited.
    Single-condition behaviour is unchanged (sum of one element).
    """
    import inspect
    import re
    import sys
    import textwrap

    name = "_prepare_hidden_inputs_anyres"
    if getattr(cls, "_vae_cond_multi_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    pat = re.compile(r"(num_post_image_token_gen_cond\[\s*cur_num_cond : cur_num_cond \+ num_conds\[b\]\s*\])")
    new, n = pat.subn(r"\1.sum()", src)
    assert n == 2, f"expected 2 slice sites in {name}, found {n}; upstream changed, re-check"
    ns = {}
    exec(compile(new, f"<patched {name}>", "exec"), sys.modules[cls.__module__].__dict__, ns)
    setattr(cls, name, ns[name])
    cls._vae_cond_multi_patched = True
