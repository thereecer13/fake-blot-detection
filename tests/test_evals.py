from fake_blot_detection.evals import mean_score, run_eval


def test_run_eval_scores_each_prompt():
    results = run_eval(str.upper, ["a", "b"], lambda p, o: float(o == p.upper()))
    assert [r.output for r in results] == ["A", "B"]
    assert mean_score(results) == 1.0


def test_mean_score_empty():
    assert mean_score([]) == 0.0
