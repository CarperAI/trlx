def enable_neox(neox_path):
    """ Enable gpt-neox support.
    HF Accelerate tries to import megatron-lm, but it's incompatible with gpt-neox's megatron """
    from accelerate.utils import imports

    def always_false():
        return False

    imports.is_megatron_lm_available = always_false

    import sys
    sys.path.append(neox_path)

    # Import models to register them for get_model
    import megatron
    print(f"using megatron from {megatron.__file__}")
    from trlx.model.neox_ilql_model import NeoXILQLModel

