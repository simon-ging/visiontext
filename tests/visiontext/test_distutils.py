from visiontext.distutils import (
    barrier_safe,
    get_global_rank,
    get_process_info,
    get_world_info,
    is_distributed,
    is_main_process,
    print_main,
)


def test_functions():
    print(get_process_info())
    assert get_world_info() == (0, 1)
    assert get_global_rank() == 0
    print_main("test")
    assert is_main_process()
    barrier_safe()
    assert not is_distributed()
