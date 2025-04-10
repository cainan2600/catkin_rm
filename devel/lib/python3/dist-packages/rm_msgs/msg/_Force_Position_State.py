# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from rm_msgs/Force_Position_State.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class Force_Position_State(genpy.Message):
  _md5sum = "73ff0e69e07c4dc10e08479dd9d3ff92"
  _type = "rm_msgs/Force_Position_State"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """float32[] joint
float32 force
uint16 arm_err
uint8 dof
"""
  __slots__ = ['joint','force','arm_err','dof']
  _slot_types = ['float32[]','float32','uint16','uint8']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       joint,force,arm_err,dof

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Force_Position_State, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.joint is None:
        self.joint = []
      if self.force is None:
        self.force = 0.
      if self.arm_err is None:
        self.arm_err = 0
      if self.dof is None:
        self.dof = 0
    else:
      self.joint = []
      self.force = 0.
      self.arm_err = 0
      self.dof = 0

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
      length = len(self.joint)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.Struct(pattern).pack(*self.joint))
      _x = self
      buff.write(_get_struct_fHB().pack(_x.force, _x.arm_err, _x.dof))
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
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      s = struct.Struct(pattern)
      end += s.size
      self.joint = s.unpack(str[start:end])
      _x = self
      start = end
      end += 7
      (_x.force, _x.arm_err, _x.dof,) = _get_struct_fHB().unpack(str[start:end])
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
      length = len(self.joint)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.joint.tostring())
      _x = self
      buff.write(_get_struct_fHB().pack(_x.force, _x.arm_err, _x.dof))
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
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      s = struct.Struct(pattern)
      end += s.size
      self.joint = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      _x = self
      start = end
      end += 7
      (_x.force, _x.arm_err, _x.dof,) = _get_struct_fHB().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_fHB = None
def _get_struct_fHB():
    global _struct_fHB
    if _struct_fHB is None:
        _struct_fHB = struct.Struct("<fHB")
    return _struct_fHB
