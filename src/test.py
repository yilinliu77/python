import numba as nb

@nb.njit
def str_to_int(s):
    final_index, result = len(s) - 1, 0
    for i,v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result

@nb.jit(nopython=True)
def str_to_float(v_str: str) -> float:
    dot_pos: int = 0
    candidate_str = v_str
    if v_str[0] == "-":
        candidate_str = candidate_str[1:]

    for index in range(len(candidate_str)):
        if candidate_str[index] == ".":
            dot_pos=index
    # print(dot_pos)
    if dot_pos == 0:
        raise

    result: float = 0
    for index in range(len(candidate_str)):
        if index < dot_pos:
            result += 10. ** (dot_pos - index - 1) * str_to_int(candidate_str[index])
        elif index > dot_pos:
            result += 10. ** (dot_pos - index)  * str_to_int(candidate_str[index])

    if v_str[0] == "-":
        result = result * -1
    return result