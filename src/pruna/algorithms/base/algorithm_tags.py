class AlgorithmTag:
    """Tag for algorithms."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class Quantizer(AlgorithmTag):
    """Tag for quantizers."""

    def __init__(self):
        super().__init__("quantizer")


class Pruner(AlgorithmTag):
    """Tag for pruners."""

    def __init__(self):
        super().__init__("pruner")


class Factorizer(AlgorithmTag):
    """Tag for factorizers."""

    def __init__(self):
        super().__init__("factorizer")


class Kernel(AlgorithmTag):
    """Tag for kernels."""

    def __init__(self):
        super().__init__("kernel")


class Cacher(AlgorithmTag):
    """Tag for cachers."""

    def __init__(self):
        super().__init__("cacher")


class Compiler(AlgorithmTag):
    """Tag for compilers."""

    def __init__(self):
        super().__init__("compiler")


class Batcher(AlgorithmTag):
    """Tag for batchers."""

    def __init__(self):
        super().__init__("batcher")
