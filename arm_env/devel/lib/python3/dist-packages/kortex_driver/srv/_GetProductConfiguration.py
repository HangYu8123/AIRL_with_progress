# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from kortex_driver/GetProductConfigurationRequest.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import kortex_driver.msg

class GetProductConfigurationRequest(genpy.Message):
  _md5sum = "fa3403cd5897c9698bc0fdcb2a453fbc"
  _type = "kortex_driver/GetProductConfigurationRequest"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """Empty input

================================================================================
MSG: kortex_driver/Empty
"""
  __slots__ = ['input']
  _slot_types = ['kortex_driver/Empty']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       input

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(GetProductConfigurationRequest, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.input is None:
        self.input = kortex_driver.msg.Empty()
    else:
      self.input = kortex_driver.msg.Empty()

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
      pass
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
      if self.input is None:
        self.input = kortex_driver.msg.Empty()
      end = 0
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
      pass
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
      if self.input is None:
        self.input = kortex_driver.msg.Empty()
      end = 0
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from kortex_driver/GetProductConfigurationResponse.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import kortex_driver.msg

class GetProductConfigurationResponse(genpy.Message):
  _md5sum = "98ed4d37d7247f3b94b20ec81a38583b"
  _type = "kortex_driver/GetProductConfigurationResponse"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """CompleteProductConfiguration output

================================================================================
MSG: kortex_driver/CompleteProductConfiguration

string kin
uint32 model
CountryCode country_code
string assembly_plant
string model_year
uint32 degree_of_freedom
uint32 base_type
uint32 end_effector_type
uint32 vision_module_type
uint32 interface_module_type
uint32 arm_laterality
uint32 wrist_type
uint32 brake_type
================================================================================
MSG: kortex_driver/CountryCode

uint32 identifier"""
  __slots__ = ['output']
  _slot_types = ['kortex_driver/CompleteProductConfiguration']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       output

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(GetProductConfigurationResponse, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.output is None:
        self.output = kortex_driver.msg.CompleteProductConfiguration()
    else:
      self.output = kortex_driver.msg.CompleteProductConfiguration()

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
      _x = self.output.kin
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.output.model, _x.output.country_code.identifier))
      _x = self.output.assembly_plant
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.output.model_year
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_8I().pack(_x.output.degree_of_freedom, _x.output.base_type, _x.output.end_effector_type, _x.output.vision_module_type, _x.output.interface_module_type, _x.output.arm_laterality, _x.output.wrist_type, _x.output.brake_type))
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
      if self.output is None:
        self.output = kortex_driver.msg.CompleteProductConfiguration()
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.output.kin = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.output.kin = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.output.model, _x.output.country_code.identifier,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.output.assembly_plant = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.output.assembly_plant = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.output.model_year = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.output.model_year = str[start:end]
      _x = self
      start = end
      end += 32
      (_x.output.degree_of_freedom, _x.output.base_type, _x.output.end_effector_type, _x.output.vision_module_type, _x.output.interface_module_type, _x.output.arm_laterality, _x.output.wrist_type, _x.output.brake_type,) = _get_struct_8I().unpack(str[start:end])
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
      _x = self.output.kin
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.output.model, _x.output.country_code.identifier))
      _x = self.output.assembly_plant
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.output.model_year
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_8I().pack(_x.output.degree_of_freedom, _x.output.base_type, _x.output.end_effector_type, _x.output.vision_module_type, _x.output.interface_module_type, _x.output.arm_laterality, _x.output.wrist_type, _x.output.brake_type))
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
      if self.output is None:
        self.output = kortex_driver.msg.CompleteProductConfiguration()
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.output.kin = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.output.kin = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.output.model, _x.output.country_code.identifier,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.output.assembly_plant = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.output.assembly_plant = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.output.model_year = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.output.model_year = str[start:end]
      _x = self
      start = end
      end += 32
      (_x.output.degree_of_freedom, _x.output.base_type, _x.output.end_effector_type, _x.output.vision_module_type, _x.output.interface_module_type, _x.output.arm_laterality, _x.output.wrist_type, _x.output.brake_type,) = _get_struct_8I().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_8I = None
def _get_struct_8I():
    global _struct_8I
    if _struct_8I is None:
        _struct_8I = struct.Struct("<8I")
    return _struct_8I
class GetProductConfiguration(object):
  _type          = 'kortex_driver/GetProductConfiguration'
  _md5sum = '899fbdc53f9306591e652c7c52962785'
  _request_class  = GetProductConfigurationRequest
  _response_class = GetProductConfigurationResponse
