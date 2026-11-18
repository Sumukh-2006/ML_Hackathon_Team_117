"""Reading single files out of a zip archive on a web server, by asking for
byte ranges only, so files we do not want are never downloaded.

A zip ends with a table of contents (the central directory): one entry per
file with its name, size, compression and where its bytes start. Each file's
bytes are preceded by a short local header whose length must be read before
the data can be located.
"""

import http.client
import struct
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass

from dataset import DatasetError

# NOTE: zip format constants (PKWARE APPNOTE): record signatures and the
# fixed lengths of the records that precede variable-length names.
CENTRAL_ENTRY: bytes = b"PK\x01\x02"
END_OF_DIRECTORY: bytes = b"PK\x05\x06"
LOCAL_HEADER_SIZE: int = 30
STORED: int = 0
DEFLATED: int = 8


@dataclass(frozen=True)
class Member:
    """One file in the archive, as its table-of-contents entry describes."""
    name: str
    method: int
    crc: int
    compressed_size: int
    header_offset: int


def fetch_range(url: str, first: int, last: int, attempts: int) -> bytes:
    """Bytes first..last (inclusive) of the file at `url`. A connection
    that drops mid-transfer is retried, up to `attempts` tries in all."""
    request = urllib.request.Request(url,
                                     headers={"Range": f"bytes={first}-{last}"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                block: bytes = response.read()
        except (http.client.IncompleteRead, urllib.error.URLError,
                TimeoutError) as error:
            if attempt == attempts:
                raise DatasetError(f"{url} bytes {first}-{last}: failed "
                                   f"{attempts} times: {error}") from None
            continue
        if len(block) == last - first + 1:
            return block
    raise DatasetError(f"{url} bytes {first}-{last}: short reply "
                       f"{attempts} times")


def table_of_contents(url: str,
                      tail_size: int) -> tuple[dict[str, Member], int]:
    """Every member listed in the archive's central directory, and the
    byte where that directory starts (where the last member's data ends).
    The directory must lie within the archive's last `tail_size` bytes."""
    tail: bytes = fetch_range_from_end(url, tail_size)
    end_record: int = tail.rfind(END_OF_DIRECTORY)
    if end_record < 0:
        raise DatasetError(f"{url}: no zip directory in the last "
                           f"{tail_size} bytes")
    directory_start, = struct.unpack("<I",
                                     tail[end_record + 16:end_record + 20])

    members: dict[str, Member] = {}
    at: int = tail.find(CENTRAL_ENTRY)
    while at >= 0:
        method, = struct.unpack("<H", tail[at + 10:at + 12])
        crc, compressed = struct.unpack("<II", tail[at + 16:at + 24])
        name_size, extra, comment = struct.unpack("<HHH", tail[at + 28:at + 34])
        offset, = struct.unpack("<I", tail[at + 42:at + 46])
        name: str = tail[at + 46:at + 46 + name_size].decode("utf-8")
        members[name] = Member(name, method, crc, compressed, offset)
        at = tail.find(CENTRAL_ENTRY, at + 46 + name_size + extra + comment)
    return members, directory_start


def fetch_range_from_end(url: str, size: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes=-{size}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def unpack(member: Member, block: bytes, block_start: int) -> bytes:
    """The member's content, cut from a downloaded block of the archive
    that starts at byte `block_start` and contains the whole member."""
    at: int = member.header_offset - block_start
    name_size, extra = struct.unpack("<HH", block[at + 26:at + 30])
    start: int = at + LOCAL_HEADER_SIZE + name_size + extra
    raw: bytes = block[start:start + member.compressed_size]

    if member.method == STORED:
        content: bytes = raw
    elif member.method == DEFLATED:
        content = zlib.decompress(raw, -15)
    else:
        raise DatasetError(f"{member.name}: compression method "
                           f"{member.method} is not supported")

    if zlib.crc32(content) != member.crc:
        raise DatasetError(f"{member.name}: content does not match the "
                           f"checksum in the archive's directory")
    return content
