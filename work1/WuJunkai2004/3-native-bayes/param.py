import json

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
    })
})

pixel = _json({
    'solid':    255/2,
    'hollow':   255/3*1
})

image = _json({
    'width':    28,
    'height':   28
})



__detail = {'1b': 15878, '1a': 19135, '1d': 11060, '1c': 13927, '2d': 16660, '2c': 12735, '2b': 14821, '2a': 15784, '3c': 10432, '3d': 32642, '3b': 8432, '3a': 8494, '4c': 11144, '4d': 27786, '4b': 9125, '4a': 11945, '5d': 23022, '5c': 12696, '5b': 11406, '5a': 12876, '6a': 28394, '6b': 15609, '6d': 5758, '6c': 10239}

probalility = _json({
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

for __key in __detail.keys():
    probalility['has_part_'+__key] = __detail[__key]/60000