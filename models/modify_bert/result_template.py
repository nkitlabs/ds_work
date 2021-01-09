import tensorflow as tf
import numpy as np

from typing import List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, fields

class ResultModel(OrderedDict):
    '''Base class for every model outputs.

    The object allows `__getitem__` to get the value not only by a key string 
    but also by an indexed integer or a slice of indexed integers or strings 
    keys that will ignore `None` attributes. 

    However, this object cannot use update/pop methods and del operation. 
    To unpack values inside object, use `to_tuple` method.
    '''
    def __post_init__(self):
        _class_fields = fields(self)
        _class_name = self.__class__.__name__


        assert len(_class_fields), f'{_class_name} has no fields.'
        assert all(
            _f.default is None for _f in _class_fields[1:]
        ), f'{_class_name} should not have more than one required field.'

        has_other_not_none_fields = any(
            getattr(self, _f.name) is not None for _f in _class_fields[1:]
        )
        
        _first_field = getattr(self, _class_fields[0].name)
        is_first_field_tensor = isinstance(_first_field, (tf.Tensor, np.ndarray)) 
        
        # in case first object is tensor or it is a pack of multiple elements,
        # assign a key value and end process.
        if is_first_field_tensor or has_other_not_none_fields:
            for _f in _class_fields:
                v = getattr(self, _f.name)
                if v is not None:
                    self[_f.name] = v 
            return
        
        # deal with the case that it is only an object by checking whether the 
        # object is iterable. If not, assign that value.
        _iterable, _iterator = False, None
        try:
            _iterator = iter(_first_field)
            _terable  = True
        except TypeError:
            _iterable = False
        
        if not _iterable:
            assert _first_field is not None, f'only field in this object is None'
            self[_class_fields[0].name] = _first_field
            return
        
        # The last case is the object is an iterable object. Iterate and assign
        # the items if it is a pair of tuple or items in list with the first one
        # is the key and the second is a value object. 
        for _elem in _iterator:
            if (
                isinstance(_elem, (list, tuple)) 
                and len(_elem) == 2 
                and isinstance(_elem[0], str)
                and _elem[1] is not None
            ):
                self[_elem[0]] = _elem[1]
    
    def to_tuple(self):
        return tuple(self[k] for k in self.keys())

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k:v for k,v in self.items()}
            return inner_dict[k]
        else:
            # in case indexing integer or list of key string or index integers. 
            return self.to_tuple()[k]
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setitem__(name, value)

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f'`__delitem__` has not been implemented on a '
            f'{self.__class__.__name__} instance.'
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f'`setdefault` has not been implemented on a '
            f'{self.__class__.__name__} instance.'
        )
    def pop(self, *args, **kwargs):
        raise Exception(
            f'`pop` has not been implemented on a {self.__class__.__name__} '
            f'instance.'
        )
    def update(self, *args, **kwargs):
        raise Exception(
            f'`update` has not been implemented on a {self.__class__.__name__} '
            f'instance.'
        )

@dataclass
class ResultBertModel(ResultModel):
    '''BERT outputs, with hidden_states and attentions, if any.
    Args:
        output: tensor object with size [batch_size, seq_length, hidden_size]
        hidden_states: (Optional) tuple of float tensors.
            All hidden states calculated from sublayers.
        attentions: (Optional) tuple of float tensors.
            Attentions weights after the attention softmax, used to compute the 
            weighted average in the self-attention heads.
    '''

    output: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
