import json
import dblite

class _json:
    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key == 'data':
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __str__(self) -> str:
        return json.dumps(self.data)


threshold = _json({
    'pixel':    96,
    'ratio':    _json({
        'width':    2,
        'height':   3
    }),
    'split':    _json({
        'width':    2,
        'height':   3
    }),
    'density':  [0.125] * 8
})

threshold['word'] = [ "{}{}".format(i,chr(97+j)) for j in range(len(threshold.density)) for i in range(1, 1 + threshold.split.width * threshold.split.height)  ]

pixel = _json({
    'solid':    255/2,
    'hollow':   255/3*1
})

image = _json({
    'width':    28,
    'height':   28
})


probability = _json({
    'is_number_0':  5923/60000,
    'is_number_1':  6742/60000,
    'is_number_2':  5958/60000,
    'is_number_3':  6131/60000,
    'is_number_4':  5842/60000,
    'is_number_5':  5421/60000,
    'is_number_6':  5918/60000,
    'is_number_7':  6265/60000,
    'is_number_8':  5851/60000,
    'is_number_9':  5949/60000
})


def other_probability():
    db = dblite.SQL("clean.db")
    try:
        db["num_1"]["part_1"]
    except:
        return

    data = {}
    for num in range(10):
        name = "num_{}".format(num)
        for i in range(1,7):
            part = "part_{}".format(i)
            for item in db[name][part]:
                if item not in data.keys():
                    data[item] = 0
                data[item] += 1
    
    print(data)

    for __key in data.keys():
        probability['has_part_'+__key] = data[__key]/60000