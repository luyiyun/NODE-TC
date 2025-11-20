from typing import Protocol, TypeVar

# 定义一个类型变量 T_co，用于表示 __getitem__ 的返回值类型。
# covariant=True 表示这个类型是协变的，这对于容器类型是常见的。
T_co = TypeVar("T_co", covariant=True)


class SizedAndGettable(Protocol[T_co]):
    """
    一个协议，用于描述任何支持通过整数索引进行访问 (__getitem__)
    并且可以获取其长度 (__len__) 的对象。
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T_co:
        # 这里的 index 通常是 int，但为了更通用，也可以是 slice。
        # 这里我们以 int 为例。
        ...
