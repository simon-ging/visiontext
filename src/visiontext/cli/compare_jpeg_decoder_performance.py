from pathlib import Path
from timeit import default_timer

from visiontext.examples import cc0images
from visiontext.images import JPEGDecoderConst, decode_jpeg

image_shapes_highres = [(5184, 3456, 3), (2038, 3057, 3), (2560, 2048, 3)]
image_shapes_lowres = [(500, 333, 3), (333, 500, 3), (500, 400, 3)]


def main():
    src_dir_base = Path(cc0images.__file__).parent

    for src_dir, image_shapes, n_trials in [
        (src_dir_base, image_shapes_highres, 2),
        (src_dir_base / "lowres", image_shapes_lowres, 100),
    ]:
        print(f"Test images in {src_dir}")
        images_f = list(sorted(src_dir.glob("*.jpg")))

        if len(images_f) != len(image_shapes):
            raise ValueError(
                f"Expected {len(image_shapes)} images in {src_dir}, got {len(images_f)}"
            )
        images_b = [Path(image_f).read_bytes() for image_f in images_f]

        for decoder in JPEGDecoderConst.values():
            if decoder == JPEGDecoderConst.PILLOW_IMAGE:
                continue
            t0 = default_timer()
            for n in range(n_trials):
                for i, image_b in enumerate(images_b):
                    arr = decode_jpeg(image_b, method=decoder)
                    if n == 0:
                        # only test shape for first trial to not disturb the performance test
                        ex_s = image_shapes[i]
                        act_s = arr.shape
                        if ex_s != act_s:
                            raise ValueError(
                                f"Expected shape {ex_s} for {images_f[i]}, got {act_s}"
                            )
            tdelta = default_timer() - t0
            tper_ms = (tdelta / n_trials / len(images_f)) * 1000
            print(f"    Decoder {decoder} took {tper_ms:.2f} ms per image on average")


if __name__ == "__main__":
    main()
