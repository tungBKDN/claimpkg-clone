from typing import Dict, List, Tuple

def generate_triplets(data, remove_underscore=False):
    """
    Tạo triplets pseudo-subgraph tổng quát từ dict chứa 'Evidence'.
    - Xử lý quan hệ bình thường, quan hệ đảo ngược (~relation)
    - Tạo placeholder unknown_i cho các tail chưa xác định (question/existence claim)
    - remove_underscore=True: thay underscore bằng space cho entity names (không đổi unknown_i)

    Args:
        data (dict): dict chứa 'Entity_set' và 'Evidence'
        remove_underscore (bool): True để thay '_' bằng ' ', False giữ nguyên

    Returns:
        dict: dict với key 'triplet' là list of (head, relation, tail)
    """
    triplets = []
    unknown_count = 0
    entity_set = set(data.get('Entity_set', []))

    def std_entity(e):
        """Chuẩn hóa entity name theo remove_underscore, giữ unknown_i nguyên"""
        if e.startswith("unknown_"):
            return e
        return e.replace("_", " ") if remove_underscore else e

    # chuẩn hóa entity_set
    entity_set = {std_entity(e) for e in entity_set}

    for entity, rel_lists in data.get('Evidence', {}).items():
        entity_std = std_entity(entity)
        for rel_list in rel_lists:
            for r in rel_list:
                if r.startswith('~'):
                    relation = r[1:]
                    tails = [std_entity(e) for e in entity_set if e != entity_std]
                    if not tails:
                        tail = f'unknown_{unknown_count}'
                        unknown_count += 1
                        # triplets.append((, relation, ))
                        triplets.append((entity_std, relation, tail))
                    else:
                        for tail in tails:
                            triplets.append((entity_std, relation, tail))
                else:
                    tails = [std_entity(e) for e in entity_set if e != entity_std]
                    if not tails:
                        tail = f'unknown_{unknown_count}'
                        unknown_count += 1
                        triplets.append((entity_std, r, tail))
                    else:
                        for tail in tails:
                            triplets.append((entity_std, r, tail))

    data['triplet'] = triplets
    return data

def generate_claimpkg_triplets(sample,
                               remove_underscore=False,
                               unknown_prefix="unknown_"):
    entity_set = sample["Entity_set"]
    evidence = sample["Evidence"]

    def norm(e):
        if remove_underscore and isinstance(e, str):
            return e.replace("_", " ")
        return e

    # standardize names
    entity_set_std = [norm(e) for e in entity_set]

    # unknown node mapping
    unk_map = {}
    unk_counter = 0

    def new_unknown():
        nonlocal unk_counter
        u = f"{unknown_prefix}{unk_counter}"
        unk_counter += 1
        return u

    def resolve_entity(e):
        if e in entity_set:
            return norm(e)
        # any implicit node labeled in evidence but not in entity_set becomes unknown
        if e not in unk_map:
            unk_map[e] = new_unknown()
        return unk_map[e]

    triplets = []

    # CASE 1: Only one entity in the claim
    if len(entity_set) == 1:
        head = entity_set_std[0]
        tail = new_unknown()  # always unknown_0
        for rel_lists in evidence.values():
            for rel_group in rel_lists:
                for r in rel_group:
                    if r.startswith("~"):
                        triplets.append((tail, r[1:], head))
                    else:
                        triplets.append((head, r, tail))
        sample['triplet'] = triplets
        return sample

    # CASE 2+: Two or more entities
    # Build unknown map for implicit nodes
    explicit = set(entity_set)
    implicit = set(evidence.keys()) - explicit
    for imp in implicit:
        resolve_entity(imp)

    # All nodes available
    all_nodes = [norm(e) for e in entity_set] + list(unk_map.values())

    # generate triplets
    for ent, rel_lists in evidence.items():
        head = resolve_entity(ent)
        # tails = every other node
        tails = [t for t in all_nodes if t != head]

        for rel_group in rel_lists:
            for r in rel_group:
                if r.startswith("~"):  # inverse
                    rel = r[1:]
                    for t in tails:
                        triplets.append((t, rel, head))
                else:                 # forward
                    for t in tails:
                        triplets.append((head, r, t))

    # dedupe
    triplets = list(dict.fromkeys(triplets))
    sample['triplet'] = triplets
    return sample


def process_data(data: dict, remove_underscore: bool = True) -> Tuple[Dict, List]:
    from tqdm import tqdm
    """
    Create triplets from given FactKG structure.

    Parameters:
    - data (dict): Input data containing 'Entity_set' and 'Evidence'.
    - remove_underscore (bool): If True, replace underscores with spaces in entity names.

    Returns:
    - Tuple[Dict, List]: A tuple containing the updated data dictionary and the list distinct entity used for later update the trie.
    """
    updated_data = {}
    distinct_entities = set()
    keys = list(data.keys())
    for key in tqdm(keys, desc="Processing data"):
        updated = generate_claimpkg_triplets(data[key], remove_underscore)
        updated_data[key] = updated

        # Collect distinct entities from all triplets
        for triplet in updated["triplet"]:
            # Triplet contains 3 elements, get the first one and the last one as entities
            distinct_entities.add(triplet[0])
            distinct_entities.add(triplet[2])

    return updated_data, list(distinct_entities)