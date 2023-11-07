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