from speechtotext.core.finder import (
    normalize, search, cluster_regions, clip_window, Region,
)


def _seg(s, e, t):
    return {"start": s, "end": e, "text": t}


def test_normalize_strips_accents_and_case():
    assert normalize("Sísmica ÑOÑO") == "sismica nono"


def test_search_groups_contiguous_into_one_region():
    segs = [
        _seg(0, 1, "hola"),
        _seg(1, 2, "la vulnerabilidad sismica"),
        _seg(2, 3, "sismica otra vez"),
        _seg(500, 501, "nada"),
    ]
    regs = search(segs, "vulnerabilidad sismica", gap=60, top=5)
    assert len(regs) == 1
    assert regs[0].start == 1 and regs[0].end == 3
    assert regs[0].hits == 2


def test_search_splits_on_large_gap():
    segs = [_seg(0, 1, "sismica"), _seg(200, 201, "sismica")]  # hueco 199 > 60
    regs = search(segs, "sismica", gap=60, top=5)
    assert len(regs) == 2


def test_search_ranks_denser_region_first():
    segs = [
        _seg(0, 1, "sismica"),
        _seg(500, 501, "sismica"), _seg(501, 502, "sismica"), _seg(502, 503, "sismica"),
    ]
    regs = search(segs, "sismica", gap=60, top=5)
    assert regs[0].start == 500  # la región densa va primero


def test_search_accent_and_case_insensitive():
    regs = search([_seg(0, 1, "la SÍSMICA de hoy")], "sismica", gap=60, top=5)
    assert len(regs) == 1


def test_search_no_match_is_empty():
    assert search([_seg(0, 1, "hola")], "inexistente", gap=60, top=5) == []


def test_clip_window_clamps_and_pads():
    assert clip_window(100.0, 160.0, 10.0) == (90.0, 80.0)
    assert clip_window(5.0, 15.0, 10.0) == (0.0, 30.0)  # no baja de 0
