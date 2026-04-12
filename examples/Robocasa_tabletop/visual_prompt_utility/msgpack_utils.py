# Shared msgpack utilities for numpy array serialization
# Used by client/server communication for SAM3 and VLM services

import numpy as np
import msgpack


def pack_array(obj):
    """Serialize numpy arrays and generics to msgpack-compatible format."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj):
    """Deserialize msgpack data back to numpy arrays and generics."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj):
    """Pack object to msgpack bytes with numpy support."""
    return msgpack.packb(obj, default=pack_array)


def unpackb(data):
    """Unpack msgpack bytes with numpy support."""
    return msgpack.unpackb(data, object_hook=unpack_array)

