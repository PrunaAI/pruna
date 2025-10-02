Tensor Annotations with jaxtyping
=================================

Purpose
-------

In this project, we use `jaxtyping` (formerly `torchtyping`) to annotate PyTorch tensors with both **shape** and **dtype** information. 

The main benefits are:

- **Improved Readability:** Developers can immediately see the expected shape and type of tensors without digging through code.
- **Early Error Detection:** Shape or dtype mismatches can be caught early during development, preventing runtime errors.
- **Self-Documenting Code:** Functions become more descriptive, reducing the need for extensive external documentation.

---

How and Where We Use It
-----------------------

`jaxtyping` is primarily used in **collate functions, dataloaders, and model input/output functions**. For example, in `image_classification_collate`:

.. code-block:: python

    from jaxtyping import Float, Int
    import torch
    from typing import Any, Tuple

    def image_classification_collate(
        data: Any, img_size: int, output_format: str = "int"
    ) -> Tuple[Float[torch.Tensor, "batch 3 height width"], Int[torch.Tensor, "batch"]]:
        """
        Collates a batch of images and labels with shape and dtype annotations.
        """
        transformations = image_format_to_transforms(output_format, img_size)
        images, labels = [], []

        for item in data:
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_tensor = transformations(image)
            images.append(image_tensor)
            labels.append(item["label"])

        images_tensor = torch.stack(images).float()
        labels_tensor = torch.tensor(labels)

        return images_tensor, labels_tensor

**Explanation:**

- `Float[torch.Tensor, "batch 3 height width"]` → `images_tensor`  
  Indicates a float tensor with `batch` size, 3 color channels, and `height x width` dimensions.
- `Int[torch.Tensor, "batch"]` → `labels_tensor`  
  Indicates an integer tensor with `batch` elements representing class labels.

---

Optional Runtime Validation
---------------------------

You can optionally validate shapes and dtypes at runtime using `check_type`:

.. code-block:: python

    from jaxtyping import check_type

    check_type(images_tensor, Float[torch.Tensor, "batch 3 height width"])
    check_type(labels_tensor, Int[torch.Tensor, "batch"])

If the tensor does not match the annotated shape/dtype, an exception will be raised immediately.

---

Where Else to Use jaxtyping
---------------------------

While we currently use it in collate functions, it is highly recommended for:

- **Model forward functions:** Annotate input and output tensors to ensure layer compatibility.
- **Loss functions:** Confirm that predictions and targets have matching shapes.
- **Custom Dataset transformations:** Validate intermediate tensors to prevent silent bugs.

---

Best Practices
--------------

- Only annotate tensors that have **fixed or predictable shapes**.
- Use descriptive dimension names like `batch`, `height`, `width`, `channels`.
- Combine with runtime checks (`check_type`) during development for safety.
- Remember that type annotations do **not affect runtime** unless you explicitly call `check_type`.

---
