# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: masking_analysis/protos/sound_generation.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='masking_analysis/protos/sound_generation.proto',
  package='sound',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n.masking_analysis/protos/sound_generation.proto\x12\x05sound\"\xc9\x01\n\x0eSoundGenConfig\x12\n\n\x02\x66s\x18\x01 \x01(\x03\x12\x10\n\x08\x64uration\x18\x02 \x01(\x03\x12+\n\x10pure_tone_config\x18\x03 \x01(\x0b\x32\x0f.sound.PureToneH\x00\x12>\n\x1a\x66lat_spectrum_noise_config\x18\x04 \x01(\x0b\x32\x18.sound.FlatSpectrumNoiseH\x00\x12$\n\x0c\x63hirp_config\x18\x05 \x01(\x0b\x32\x0c.sound.ChirpH\x00\x42\x06\n\x04type\"\x1f\n\x08PureTone\x12\x13\n\x0b\x63\x65nter_freq\x18\x01 \x01(\x03\"\xa9\x01\n\x05\x43hirp\x12\x12\n\nstart_freq\x18\x01 \x01(\x03\x12\x11\n\tstop_freq\x18\x02 \x01(\x03\x12.\n\x0csweep_method\x18\x03 \x01(\x0e\x32\x18.sound.Chirp.SweepMethod\"I\n\x0bSweepMethod\x12\n\n\x06LINEAR\x10\x00\x12\r\n\tQUADRATIC\x10\x01\x12\x0f\n\x0bLOGARITHMIC\x10\x02\x12\x0e\n\nHYPERBOLIC\x10\x03\"P\n\x11\x46latSpectrumNoise\x12\x14\n\x0c\x66ilter_order\x18\x03 \x01(\x03\x12\x12\n\nstart_freq\x18\x01 \x01(\x03\x12\x11\n\tstop_freq\x18\x02 \x01(\x03\x62\x06proto3')
)



_CHIRP_SWEEPMETHOD = _descriptor.EnumDescriptor(
  name='SweepMethod',
  full_name='sound.Chirp.SweepMethod',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LINEAR', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='QUADRATIC', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOGARITHMIC', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HYPERBOLIC', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=391,
  serialized_end=464,
)
_sym_db.RegisterEnumDescriptor(_CHIRP_SWEEPMETHOD)


_SOUNDGENCONFIG = _descriptor.Descriptor(
  name='SoundGenConfig',
  full_name='sound.SoundGenConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='fs', full_name='sound.SoundGenConfig.fs', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='duration', full_name='sound.SoundGenConfig.duration', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pure_tone_config', full_name='sound.SoundGenConfig.pure_tone_config', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flat_spectrum_noise_config', full_name='sound.SoundGenConfig.flat_spectrum_noise_config', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='chirp_config', full_name='sound.SoundGenConfig.chirp_config', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='type', full_name='sound.SoundGenConfig.type',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=58,
  serialized_end=259,
)


_PURETONE = _descriptor.Descriptor(
  name='PureTone',
  full_name='sound.PureTone',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='center_freq', full_name='sound.PureTone.center_freq', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=261,
  serialized_end=292,
)


_CHIRP = _descriptor.Descriptor(
  name='Chirp',
  full_name='sound.Chirp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start_freq', full_name='sound.Chirp.start_freq', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stop_freq', full_name='sound.Chirp.stop_freq', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sweep_method', full_name='sound.Chirp.sweep_method', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _CHIRP_SWEEPMETHOD,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=295,
  serialized_end=464,
)


_FLATSPECTRUMNOISE = _descriptor.Descriptor(
  name='FlatSpectrumNoise',
  full_name='sound.FlatSpectrumNoise',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filter_order', full_name='sound.FlatSpectrumNoise.filter_order', index=0,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_freq', full_name='sound.FlatSpectrumNoise.start_freq', index=1,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stop_freq', full_name='sound.FlatSpectrumNoise.stop_freq', index=2,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=466,
  serialized_end=546,
)

_SOUNDGENCONFIG.fields_by_name['pure_tone_config'].message_type = _PURETONE
_SOUNDGENCONFIG.fields_by_name['flat_spectrum_noise_config'].message_type = _FLATSPECTRUMNOISE
_SOUNDGENCONFIG.fields_by_name['chirp_config'].message_type = _CHIRP
_SOUNDGENCONFIG.oneofs_by_name['type'].fields.append(
  _SOUNDGENCONFIG.fields_by_name['pure_tone_config'])
_SOUNDGENCONFIG.fields_by_name['pure_tone_config'].containing_oneof = _SOUNDGENCONFIG.oneofs_by_name['type']
_SOUNDGENCONFIG.oneofs_by_name['type'].fields.append(
  _SOUNDGENCONFIG.fields_by_name['flat_spectrum_noise_config'])
_SOUNDGENCONFIG.fields_by_name['flat_spectrum_noise_config'].containing_oneof = _SOUNDGENCONFIG.oneofs_by_name['type']
_SOUNDGENCONFIG.oneofs_by_name['type'].fields.append(
  _SOUNDGENCONFIG.fields_by_name['chirp_config'])
_SOUNDGENCONFIG.fields_by_name['chirp_config'].containing_oneof = _SOUNDGENCONFIG.oneofs_by_name['type']
_CHIRP.fields_by_name['sweep_method'].enum_type = _CHIRP_SWEEPMETHOD
_CHIRP_SWEEPMETHOD.containing_type = _CHIRP
DESCRIPTOR.message_types_by_name['SoundGenConfig'] = _SOUNDGENCONFIG
DESCRIPTOR.message_types_by_name['PureTone'] = _PURETONE
DESCRIPTOR.message_types_by_name['Chirp'] = _CHIRP
DESCRIPTOR.message_types_by_name['FlatSpectrumNoise'] = _FLATSPECTRUMNOISE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SoundGenConfig = _reflection.GeneratedProtocolMessageType('SoundGenConfig', (_message.Message,), dict(
  DESCRIPTOR = _SOUNDGENCONFIG,
  __module__ = 'masking_analysis.protos.sound_generation_pb2'
  # @@protoc_insertion_point(class_scope:sound.SoundGenConfig)
  ))
_sym_db.RegisterMessage(SoundGenConfig)

PureTone = _reflection.GeneratedProtocolMessageType('PureTone', (_message.Message,), dict(
  DESCRIPTOR = _PURETONE,
  __module__ = 'masking_analysis.protos.sound_generation_pb2'
  # @@protoc_insertion_point(class_scope:sound.PureTone)
  ))
_sym_db.RegisterMessage(PureTone)

Chirp = _reflection.GeneratedProtocolMessageType('Chirp', (_message.Message,), dict(
  DESCRIPTOR = _CHIRP,
  __module__ = 'masking_analysis.protos.sound_generation_pb2'
  # @@protoc_insertion_point(class_scope:sound.Chirp)
  ))
_sym_db.RegisterMessage(Chirp)

FlatSpectrumNoise = _reflection.GeneratedProtocolMessageType('FlatSpectrumNoise', (_message.Message,), dict(
  DESCRIPTOR = _FLATSPECTRUMNOISE,
  __module__ = 'masking_analysis.protos.sound_generation_pb2'
  # @@protoc_insertion_point(class_scope:sound.FlatSpectrumNoise)
  ))
_sym_db.RegisterMessage(FlatSpectrumNoise)


# @@protoc_insertion_point(module_scope)
