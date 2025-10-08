Tensor Annotations with jaxtyping
======================


We use `jaxtyping <https://github.com/google/jaxtyping>`_ to annotate PyTorch tensors with both shape and dtype information, mainly in collate functions, dataloaders, and model interfaces. This improves readability, helps catch shape/type mismatches early, and makes functions more self-documenting. Contributors are encouraged to follow this style when defining tensor inputs or outputs—for details and examples, see the official jaxtyping documentation.