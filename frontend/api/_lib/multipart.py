"""multipart/form-data parsing with the stdlib email parser (cgi is gone in 3.13)."""

from email.parser import BytesParser
from email.policy import HTTP


class FormError(Exception):
    pass


def parse(content_type, body):
    """-> {field: {"value": bytes, "filename": str or None}}."""
    if not content_type or not content_type.lower().startswith("multipart/form-data"):
        raise FormError("Expected a multipart/form-data upload.")
    message = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not message.is_multipart():
        raise FormError("Malformed multipart body.")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = {"value": part.get_payload(decode=True) or b"",
                            "filename": part.get_filename()}
    return fields


def text(fields, name):
    field = fields.get(name)
    if field is None:
        return None
    value = field["value"].decode("utf-8", errors="replace").strip()
    return value or None
