import torch
import os

from visiontext.images import visualize_image_text_pairs_from_tensor


def test_visualize_image_text_pairs_from_tensor(tmpdir):
    # Create a dummy tensor of shape (B, C, H, W) with values between 0-1
    B, C, H, W = 2, 3, 364, 164
    image_tensor = torch.rand(B, C, H, W) * 0.4
    image_tensor[0, 0] += 0.5  # first image red
    image_tensor[1, 1] += 0.5  # second image green

    # Create dummy text captions
    captions = ["WWWWWWWWWW WW", "Caption 2 make it longer to test text wrapping and so on."]

    # Call the visualization function
    result = visualize_image_text_pairs_from_tensor(image_tensor, captions)

    # Verify the result is a list of PIL images with the correct length
    assert len(result) == B

    # Save images to temporary directory
    for i, img in enumerate(result):
        img_path = os.path.join(tmpdir, f"image_{i}.png")
        img.save(img_path)
        print(f"Saved image {i} to: {img_path}")

    # Print the temporary directory path
    print(f"All images saved in: {tmpdir}")
