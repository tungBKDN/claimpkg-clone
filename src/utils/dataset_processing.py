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
                        triplets.append((tail, relation, entity_std))
                    else:
                        for tail in tails:
                            triplets.append((tail, relation, entity_std))
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
        updated = generate_triplets(data[key], remove_underscore)
        updated_data[key] = updated

        # Collect distinct entities from all triplets
        for triplet in updated["triplet"]:
            # Triplet contains 3 elements, get the first one and the last one as entities
            distinct_entities.add(triplet[0])
            distinct_entities.add(triplet[2])

    return updated_data, list(distinct_entities)