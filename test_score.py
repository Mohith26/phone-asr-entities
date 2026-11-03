import score


def test_dob_formats():
    assert score.norm_dob("My date of birth is July 25th, 1993.") == "1993-07-25"
    assert score.norm_dob("born March 4 1987") == "1987-03-04"
    assert score.norm_dob("it's 7/25/1993") == "1993-07-25"
    assert score.norm_dob("no date here") is None


def test_phone_digits_and_words():
    assert score.norm_phone("the best number to reach me is 834-456-8725.") == "8344568725"
    assert score.norm_phone("reach me is nine one nine, five five five, oh one four two") == "9195550142"


def test_med_name_and_dose():
    assert score.norm_med("I need a refill on my lisinopril, the 10mg tablets.") == "lisinopril|10"
    assert score.norm_med("refill on my Tanzel Austin, the 4 milligram tablets") == "tanzelaustin|4"


def test_spelled_letters():
    assert score.norm_spelled("My last name is Okafor, that's O, K, A, F, O, R.") == "okafor"


def test_mulaw_round_trip_is_close():
    import numpy as np

    from make_clips import mulaw

    x = np.linspace(-0.9, 0.9, 1001)
    assert np.max(np.abs(mulaw(x) - x)) < 0.03
