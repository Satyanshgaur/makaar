from makaar_eval.gates import check

G = {"accuracy_margin_vs_baseline": 0.03, "formula_compile_rate_min": 0.9,
     "median_seconds_per_page_max": 60, "loops_max": 0}


def res(ned, sec, loops=0):
    return {"summary": {"printed": {"ned_mean": ned, "sec_median": sec, "loops": loops}}, "pages": []}


def test_pass():
    assert check(res(0.05, 30), G, res(0.04, 5)) == []


def test_fail_speed_accuracy_loops():
    assert len(check(res(0.2, 90, 1), G, res(0.04, 5))) == 3


def test_empty_fails():
    assert check({"summary": {}, "pages": []}, G, None)
