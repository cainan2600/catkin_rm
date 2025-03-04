# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from rm_msgs/Six_Force.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class Six_Force(genpy.Message):
  _md5sum = "abfa542f676ea571474ea027ddb54a05"
  _type = "rm_msgs/Six_Force"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """float32 force_Fx
float32 force_Fy
float32 force_Fz
float32 force_Mx
float32 force_My
float32 force_Mz"""
  __slots__ = ['force_Fx','force_Fy','force_Fz','force_Mx','force_My','force_Mz']
  _slot_types = ['float32','float32','float32','float32','float32','float32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       force_Fx,force_Fy,force_Fz,force_Mx,force_My,force_Mz

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Six_Force, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.force_Fx is None:
        self.force_Fx = 0.
      if self.force_Fy is None:
        self.force_Fy = 0.
      if self.force_Fz is None:
        self.force_Fz = 0.
      if self.force_Mx is None:
        self.force_Mx = 0.
      if self.force_My is None:
        self.force_My = 0.
      if self.force_Mz is None:
        self.force_Mz = 0.
    else:
      self.force_Fx = 0.
      self.force_Fy = 0.
      self.force_Fz = 0.
      self.force_Mx = 0.
      self.force_My = 0.
      self.force_Mz = 0.

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_6f().pack(_x.force_Fx, _x.force_Fy, _x.force_Fz, _x.force_Mx, _x.force_My, _x.force_Mz))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      end = 0
      _x = self
      start = end
      end += 24
      (_x.force_Fx, _x.force_Fy, _x.force_Fz, _x.force_Mx, _x.force_My, _x.force_Mz,) = _get_struct_6f().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_6f().pack(_x.force_Fx, _x.force_Fy, _x.force_Fz, _x.force_Mx, _x.force_My, _x.force_Mz))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      end = 0
      _x = self
      start = end
      end += 24
      (_x.force_Fx, _x.force_Fy, _x.force_Fz, _x.force_Mx, _x.force_My, _x.force_Mz,) = _get_struct_6f().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_6f = None
def _get_struct_6f():
    global _struct_6f
    if _struct_6f is None:
        _struct_6f = struct.Struct("<6f")
    return _struct_6f
