import os
import sys
import re
import duckdb
import json

sys.path.insert(0, os.path.dirname(__file__))
import normalization as norm

def main():
    con = duckdb.connect()

    # Query 10,000 ground truth pairs
    query = """
        WITH gt_sample AS (
            SELECT source1_entity_id, string_split(matched_entity_ids, ',') as id_list
            FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, all_varchar=true)
            WHERE LEN(matched_entity_ids) > 0
            USING SAMPLE 10000 (reservoir, 42)
        ),
        unnested AS (
            SELECT source1_entity_id, UNNEST(id_list) as matched_id
            FROM gt_sample
        ),
        s2_pairs AS (
            SELECT source1_entity_id, matched_id, 'S2' as source
            FROM unnested WHERE matched_id LIKE 'S2-%'
            LIMIT 5000
        ),
        s3_pairs AS (
            SELECT source1_entity_id, matched_id, 'S3' as source
            FROM unnested WHERE matched_id LIKE 'S3-%'
            LIMIT 5000
        ),
        all_pairs AS (
            SELECT * FROM s2_pairs UNION ALL SELECT * FROM s3_pairs
        )
        SELECT 
            p.source1_entity_id as s1_id,
            s1.business_name as s1_name,
            s1.business_address as s1_addr,
            s1.country as s1_country,
            p.matched_id,
            p.source,
            COALESCE(s2.business_name, s3.business_name) as m_name,
            COALESCE(s2.business_address, s3.business_address) as m_addr,
            COALESCE(s2.country, s3.country) as m_country
        FROM all_pairs p
        JOIN read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, all_varchar=true) s1
            ON p.source1_entity_id = s1.entity_id
        LEFT JOIN read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, all_varchar=true) s2
            ON p.matched_id = s2.entity_id
        LEFT JOIN read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, all_varchar=true) s3
            ON p.matched_id = s3.entity_id;
    """

    pairs = con.execute(query).fetchall()

    HONORIFICS = re.compile(r'\b(?:sri|shri|smt|m/s|ms|messrs)\b', re.IGNORECASE)

    def clean_enhanced(name):
        c = norm.clean_business_name(name)
        c = HONORIFICS.sub(' ', c)
        return re.sub(r'\s+', ' ', c).strip()

    def sorted_2_tokens(text):
        tokens = text.split()
        if len(tokens) >= 2:
            return ' '.join(sorted(tokens[:2]))
        return text

    hits = {'S2': 0, 'S3': 0}

    for row in pairs:
        s1_id, s1_name, s1_addr, s1_c, m_id, src, m_name, m_addr, m_c = row
        m_name = m_name or ''
        m_addr = m_addr or ''
        
        # Enhanced normalization
        c1 = clean_enhanced(s1_name)
        cm = clean_enhanced(m_name)
        r1 = norm.get_root_name(c1)
        rm = norm.get_root_name(cm)
        
        ca1 = norm.clean_business_address(s1_addr)
        cam = norm.clean_business_address(m_addr)
        
        nums1 = norm.extract_numeric_tokens(ca1)
        numsm = norm.extract_numeric_tokens(cam)
        num1 = nums1[0] if nums1 else ''
        numm = numsm[0] if numsm else ''
        
        # Key A: exact root name + country
        kA_1 = r1 + '||' + s1_c
        kA_m = rm + '||' + m_c
        match_A = bool(r1 and kA_1 == kA_m)
        
        # Key B: sorted first 2 root tokens + country (pure name, works even when address is missing!)
        s_tok1 = sorted_2_tokens(r1)
        s_tokm = sorted_2_tokens(rm)
        kB_1 = s_tok1 + '||' + s1_c
        kB_m = s_tokm + '||' + m_c
        match_B = bool(s_tok1 and len(s_tok1.split()) >= 2 and kB_1 == kB_m)
        
        # Key C: first 2 tokens + first num token + country
        tok1 = norm.get_first_n_tokens(r1, 2)
        tokm = norm.get_first_n_tokens(rm, 2)
        kC_1 = tok1 + '||' + num1 + '||' + s1_c
        kC_m = tokm + '||' + numm + '||' + m_c
        match_C = bool(tok1 and num1 and kC_1 == kC_m)
        
        # Key D: 6 char prefix of root name + first num + country
        p6_1 = re.sub(r'[^\w]', '', r1)[:6]
        p6_m = re.sub(r'[^\w]', '', rm)[:6]
        kD_1 = p6_1 + '||' + num1 + '||' + s1_c
        kD_m = p6_m + '||' + numm + '||' + m_c
        match_D = bool(len(p6_1) >= 4 and num1 and kD_1 == kD_m)
        
        # Key E: Numeric address match for cross-script transliteration
        common_nums = set(nums1).intersection(set(numsm))
        # Non-ASCII or same first 2 chars
        match_E = bool(common_nums and (s_tok1[:2] == s_tokm[:2] or not s1_name.isascii() or not m_name.isascii()))
        
        if match_A or match_B or match_C or match_D or match_E:
            hits[src] += 1

    s2_cov = hits['S2'] / 5000 * 100
    s3_cov = hits['S3'] / 5000 * 100
    tot_cov = (hits['S2'] + hits['S3']) / 10000 * 100

    print("Enhanced Blocking Coverage on True Positives:")
    print(f"  S2 Coverage : {hits['S2']}/5000 ({s2_cov:.2f}%)  [was 73.90%]")
    print(f"  S3 Coverage : {hits['S3']}/5000 ({s3_cov:.2f}%)  [was 65.86%]")
    print(f"  Overall     : {(hits['S2']+hits['S3'])}/10000 ({tot_cov:.2f}%)  [was 69.88%]")

if __name__ == "__main__":
    main()
