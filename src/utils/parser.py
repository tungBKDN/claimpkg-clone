from typing import Tuple

def str_to_triplet(s: str) -> Tuple[str, str, str]:
    """
    Parse a single ERE (entity-relation-entity) string into a 3-tuple: (subject, relation, object).

    Accepted entity forms:
      - Tagged: "<e>Entity Name</e>"  (tags are removed and inner text is stripped)
      - Raw token: "unknown_0" (returned as-is, trimmed)

    Relation rules:
      - May contain spaces or underscores.
      - May be negated with a leading '~'. Any spaces between '~' and the relation are removed
        so negation is normalized (e.g. "~   birth place" -> "~birth place").
      - Internal whitespace is collapsed to a single space.

    Examples:
      "<e>Ent1</e> || relation || <e>Ent2</e>" -> ("Ent1", "relation", "Ent2")
      "unknown_0 || ~birth place || <e>Vedat Tek</e>" -> ("unknown_0", "~birth place", "Vedat Tek")

    Raises:
      ValueError: if the input does not split into exactly three parts using '||'.
    """
    import re

    def _extract_entity(part: str) -> str:
        # If part contains <e>...</e>, return inner text; otherwise return trimmed token.
        m = re.search(r'<e>\s*(.*?)\s*</e>', part, flags=re.IGNORECASE)
        return m.group(1) if m else part.strip()

    parts = [p.strip() for p in s.split('||')]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 parts separated by '||', got {len(parts)}: {s!r}")

    ent1 = _extract_entity(parts[0])
    relation = parts[1]
    # Collapse internal whitespace and trim
    relation = re.sub(r'\s+', ' ', relation).strip()
    # Normalize leading negation: ensure '~' is attached to the relation token (no spaces after '~')
    relation = re.sub(r'^\~\s*', '~', relation)

    ent2 = _extract_entity(parts[2])
    return ent1, relation, ent2