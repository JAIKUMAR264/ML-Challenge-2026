import os
import sys
import duckdb
import json

sys.path.insert(0, os.path.dirname(__file__))
import normalization as norm

def main():
    con = duckdb.connect()
    con.execute("SET max_memory = '3.0GB'")
    con.execute("SET threads = 4")

    print("Sampling 10,000 ground truth pairs (5,000 S2 and 5,000 S3)...")
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
    print(f"Retrieved {len(pairs)} pairs for evaluation.")

    stats = {
        'S2': {'total': 0, 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'any': 0, 'missed': 0},
        'S3': {'total': 0, 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'any': 0, 'missed': 0}
    }

    missed_samples = {'S2': [], 'S3': []}

    for row in pairs:
        s1_id, s1_name, s1_addr, s1_c, m_id, src, m_name, m_addr, m_c = row
        m_name = m_name or ''
        m_addr = m_addr or ''

        # Keys for S1
        c1_name = norm.clean_business_name(s1_name)
        r1_name = norm.get_root_name(c1_name)
        c1_addr = norm.clean_business_address(s1_addr)
        k1_dict = norm.generate_blocking_keys(r1_name, c1_name, s1_addr, c1_addr, s1_c)

        # Keys for Match
        cm_name = norm.clean_business_name(m_name)
        rm_name = norm.get_root_name(cm_name)
        cm_addr = norm.clean_business_address(m_addr)
        km_dict = norm.generate_blocking_keys(rm_name, cm_name, m_addr, cm_addr, m_c)

        m_p1 = bool(k1_dict['pass1'] and k1_dict['pass1'] == km_dict['pass1'])
        m_p2 = bool(k1_dict['pass2'] and k1_dict['pass2'] == km_dict['pass2'])
        m_p3 = bool(k1_dict['pass3'] and k1_dict['pass3'] == km_dict['pass3'])
        m_p4 = bool(k1_dict['pass4'] and k1_dict['pass4'] == km_dict['pass4'])

        m_any = m_p1 or m_p2 or m_p3 or m_p4

        stats[src]['total'] += 1
        if m_p1: stats[src]['p1'] += 1
        if m_p2: stats[src]['p2'] += 1
        if m_p3: stats[src]['p3'] += 1
        if m_p4: stats[src]['p4'] += 1
        if m_any:
            stats[src]['any'] += 1
        else:
            stats[src]['missed'] += 1
            if len(missed_samples[src]) < 10:
                missed_samples[src].append({
                    's1_name': s1_name,
                    'm_name': m_name,
                    's1_addr': s1_addr,
                    'm_addr': m_addr,
                    's1_c': s1_c,
                    'k1_s1': k1_dict,
                    'k1_m': km_dict
                })

    for src in ['S2', 'S3']:
        tot = stats[src]['total']
        print(f"\n=== STATS FOR {src} (Total: {tot}) ===")
        print(f"  Pass 1 (exact root name)       : {stats[src]['p1']:5d} ({stats[src]['p1']/tot*100:6.2f}%)")
        print(f"  Pass 2 (2 tokens + num addr)   : {stats[src]['p2']:5d} ({stats[src]['p2']/tot*100:6.2f}%)")
        print(f"  Pass 3 (6 chars + 3 region)    : {stats[src]['p3']:5d} ({stats[src]['p3']/tot*100:6.2f}%)")
        print(f"  Pass 4 (num addr + 3 char name): {stats[src]['p4']:5d} ({stats[src]['p4']/tot*100:6.2f}%)")
        print(f"  Union of all 4 passes          : {stats[src]['any']:5d} ({stats[src]['any']/tot*100:6.2f}%)")
        print(f"  Missed by all 4 passes         : {stats[src]['missed']:5d} ({stats[src]['missed']/tot*100:6.2f}%)")

    report_path = "student_resource/outputs/step3_baseline/reports/blocking_pass_breakdown.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump({
            "stats": stats,
            "missed_samples": missed_samples
        }, fp, indent=2, ensure_ascii=False)
    print(f"\nDetailed diagnostics saved to {report_path}")

if __name__ == "__main__":
    main()
