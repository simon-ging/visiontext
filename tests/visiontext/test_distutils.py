from visiontext.distutils import (
    get_process_info,
    get_world_info,
    get_global_rank,
    print_main,
    is_main_process,
    barrier_safe,
    is_distributed,
)


def test_functions():
    print(get_process_info())
    assert get_world_info() == (0, 1)
    assert get_global_rank() == 0
    print_main("test")
    assert is_main_process()
    barrier_safe()
    assert not is_distributed()
