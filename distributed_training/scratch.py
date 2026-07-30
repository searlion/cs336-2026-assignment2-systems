def test(**kwargs):
    for key, value in kwargs.items():
        print(key, value)

test(a=1, b=2)