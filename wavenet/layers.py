# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py

from tensorflow import keras
from tensorflow.keras import backend as K



class AddSingletonDepth(keras.layers.Layer):

    def call(self, x):
        x = K.expand_dims(x, -1)  # add a dimension of the right

        if keras.backend.ndim(x) == 4:
            return K.permute_dimensions(x, (0, 3, 1, 2))
        else:
            return x

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Subtract(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, x):
        return x[0] - x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# class Slice(keras.layers.Layer):

#     def __init__(self, selector, output_shape, **kwargs):
#         self.selector = selector
#         self.desired_output_shape = output_shape
#         super(Slice, self).__init__(**kwargs)

#     def call(self, x):

#         selector = self.selector
#         if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
#             x = K.permute_dimensions(x, [0, 2, 1])
#             selector = (self.selector[1], self.selector[0])

#         y = x[selector]

#         if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
#             y = K.permute_dimensions(y, [0, 2, 1])

#         return y

#     def compute_output_shape(self, input_shape):

#         output_shape = (None,)
#         for i, dim_length in enumerate(self.desired_output_shape):
#             if dim_length == Ellipsis:
#                 output_shape = output_shape + (input_shape[i+1],)
#             else:
#                 output_shape = output_shape + (dim_length,)
#         return output_shape

import tensorflow.keras.backend as K
from tensorflow import keras

class Slice(keras.layers.Layer):
    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)

    def call(self, x):
        selector = self.selector
        if (
            len(self.selector) == 2
            and not isinstance(self.selector[1], (slice, int))
        ):
            x = K.permute_dimensions(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if (
            len(self.selector) == 2
            and not isinstance(self.selector[1], (slice, int))
        ):
            y = K.permute_dimensions(y, [0, 2, 1])

        return y

    def compute_output_shape(self, input_shape):
        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape

    def get_config(self):
        config = super().get_config()
        # Convert slice and Ellipsis into serializable form
        config.update({
            "selector": self._serialize_selector(self.selector),
            "output_shape": self._serialize_shape(self.desired_output_shape),
        })
        return config

    @classmethod
    def from_config(cls, config):
        selector = cls._deserialize_selector(config.pop("selector"))
        output_shape = cls._deserialize_shape(config.pop("output_shape"))
        return cls(selector, output_shape, **config)

    @staticmethod
    def _serialize_selector(selector):
        return [
            f"slice({s.start},{s.stop},{s.step})" if isinstance(s, slice) else s
            for s in selector
        ]

    @staticmethod
    def _deserialize_selector(serialized):
        result = []
        for s in serialized:
            if isinstance(s, str) and s.startswith("slice"):
                # Extract slice values
                parts = s[s.index("(")+1 : s.index(")")].split(",")
                start, stop, step = (int(p) if p != 'None' else None for p in parts)
                result.append(slice(start, stop, step))
            else:
                result.append(s)
        return tuple(result)

    @staticmethod
    def _serialize_shape(shape):
        return [":ellipsis:" if s is Ellipsis else s for s in shape]

    @staticmethod
    def _deserialize_shape(shape):
        return [Ellipsis if s == ":ellipsis:" else s for s in shape]

