"""Lightweight PDK shim for structure recognition only.

The full FinFET14nm_Mock_PDK imports layout generator modules.  Structure
recognition only needs models.sp and template annotations, so this generated
package intentionally exposes no layout generators.
"""


def generator_class(name):
    return False
