"""Small Draft 2020-12 validator for the two checked-in result schemas.

The cluster and the local Python installation do not require jsonschema. This
supports exactly the keywords used by the approved bench.1/ncu.1 schemas.
"""
from __future__ import annotations

import re


def validate(value, schema, *, root=None, path='$'):
    root = root or schema
    if '$ref' in schema:
        node = root
        for key in schema['$ref'].lstrip('#/').split('/'):
            node = node[key]
        return validate(value, node, root=root, path=path)
    if 'oneOf' in schema:
        matches = 0
        for sub in schema['oneOf']:
            try:
                validate(value, sub, root=root, path=path)
                matches += 1
            except ValueError:
                pass
        if matches != 1:
            raise ValueError(f'{path}: expected exactly one schema match, got {matches}')
    kind = schema.get('type')
    if kind is not None:
        kinds = [kind] if isinstance(kind, str) else kind
        predicates = {'object': lambda x: isinstance(x, dict),
                      'array': lambda x: isinstance(x, list),
                      'string': lambda x: isinstance(x, str),
                      'integer': lambda x: isinstance(x, int) and not isinstance(x, bool),
                      'number': lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
                      'boolean': lambda x: isinstance(x, bool),
                      'null': lambda x: x is None}
        if not any(predicates[k](value) for k in kinds):
            raise ValueError(f'{path}: expected {kinds}, got {type(value).__name__}')
    if 'const' in schema and value != schema['const']:
        raise ValueError(f'{path}: expected {schema["const"]!r}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: invalid enum {value!r}')
    if isinstance(value, dict):
        for key in schema.get('required', []):
            if key not in value:
                raise ValueError(f'{path}: missing {key}')
        props = schema.get('properties', {})
        additional = schema.get('additionalProperties', True)
        for key, item in value.items():
            if key in props:
                validate(item, props[key], root=root, path=f'{path}.{key}')
            elif additional is False:
                raise ValueError(f'{path}: unexpected {key}')
            elif isinstance(additional, dict):
                validate(item, additional, root=root, path=f'{path}.{key}')
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')):
            raise ValueError(f'{path}: wrong array length')
        if schema.get('uniqueItems') and len({repr(x) for x in value}) != len(value):
            raise ValueError(f'{path}: duplicate array item')
        prefix = schema.get('prefixItems', [])
        for i, item in enumerate(value):
            if i < len(prefix):
                validate(item, prefix[i], root=root, path=f'{path}[{i}]')
            elif schema.get('items') is False:
                raise ValueError(f'{path}: too many items')
            elif isinstance(schema.get('items'), dict):
                validate(item, schema['items'], root=root, path=f'{path}[{i}]')
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0):
            raise ValueError(f'{path}: string too short')
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            raise ValueError(f'{path}: does not match {schema["pattern"]}')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum']:
            raise ValueError(f'{path}: below minimum')
        if 'exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']:
            raise ValueError(f'{path}: below exclusive minimum')
