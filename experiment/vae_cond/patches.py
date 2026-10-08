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
