import json
from collections import defaultdict
from pathlib import Path


class PechaService:

    def __init__(self, pecha_id: str):
        self.mapping_dir = Path(__file__).parent / "data" / "mapping" / pecha_id

    def get_ranges(self, vol_id: str, mapping_fn: Path) -> list[tuple[str, str]]:
        mapping = json.loads(mapping_fn.read_text())
        return mapping[vol_id]

    def get_text_pages_names(self, vol_id: str) -> list[tuple[str, str]]:
        mapping_fn = self.mapping_dir / "text.json"
        return self.get_ranges(vol_id, mapping_fn)

    def get_note_pages_names(self, vol_id: str) -> list[tuple[str, str]]:
        mapping_fn = self.mapping_dir / "note.json"
        return self.get_ranges(vol_id, mapping_fn)

    def in_ranges(self, page_name: str, ranges: list[tuple[str, str]]) -> bool:
        for page_range in ranges:
            if page_range[0] <= page_name <= page_range[1]:
                return True
        return False

    def is_text_page(self, page_name: str) -> bool:
        vol_id = page_name[:-4]
        text_pages = self.get_text_pages_names(vol_id)
        return self.in_ranges(page_name, text_pages)

    def is_note_page(self, page_name: str) -> bool:
        vol_id = page_name[:-4]
        note_pages = self.get_note_pages_names(vol_id)
        return self.in_ranges(page_name, note_pages)


def convert_mapping(old_mapping_fn: Path):
    old_mapping = json.loads(old_mapping_fn.read_text())
    new_mappings = defaultdict(list)
    for group_ranges in old_mapping.values():
        group_name = group_ranges[0][:-4]
        new_mappings[group_name].append(group_ranges)
    new_mapping_fn = old_mapping_fn.parent / f"new_{old_mapping_fn.name}"
    json.dump(new_mappings, new_mapping_fn.open("w"), indent=2)


if __name__ == "__main__":
    pecha = PechaService("W1PD95844")
    print(pecha.is_text_page("I1PD958470040"))
    print(pecha.is_text_page("I1PD958470042"))
    print(pecha.is_note_page("I1PD958470049"))
